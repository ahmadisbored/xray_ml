import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import pickle
import pydicom
from PIL import Image
import nibabel as nib
import cv2
from tkinter import Tk
from tkinter.filedialog import askdirectory
from tkinter.simpledialog import askstring
import os
from dirpath import *

FINDINGS = []

data = []


def dicom_to_jpeg(file_path, file_name, save_dir, folder_name):
    ds = pydicom.dcmread(file_path)
    new_image = ds.pixel_array.astype(float)
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)
    try:
        final_image.save(os.path.join(save_dir, folder_name, (str(file_name[:-6]) + '.jpg')))
        print((str(file_name[:-6]) + '.jpg') + f' saved to {os.path.join(save_dir, folder_name)}')
    except:
        os.makedirs(os.path.join(save_dir, folder_name))
        final_image.save(os.path.join(save_dir, folder_name, (str(file_name[:-6]) + '.jpg')))
        print((str(file_name[:-6]) + '.jpg') + f' saved to {os.path.join(save_dir, folder_name)}')


def dicom_path_to_ml_model(dicom_folder_file_path, train_split, test_split, val_split):
    save_dir = askdirectory(title='Select Save Directory')
    model_folder_name = askstring(title="DICOM->ML", prompt="Name your model folder (do not use the name of a pre-existing folder):")
    for file in os.listdir(dicom_folder_file_path):
        dicom_to_jpeg(os.path.join(dicom_folder_file_path,file), file, save_dir, model_folder_name)
    final_dir = os.path.join(save_dir, model_folder_name)
    train_count = int(len(os.listdir(final_dir))*float(train_split))
    print(train_count)
    print(len(os.listdir(final_dir)))
    test_count = int(len(os.listdir(final_dir))*float(test_split))
    val_count = int(len(os.listdir(final_dir))*float(val_split))
    os.makedirs(os.path.join(final_dir, 'train'))
    os.makedirs(os.path.join(final_dir, 'test'))
    os.makedirs(os.path.join(final_dir, 'val'))
    num = 0
    for file in os.listdir(final_dir):
        if num < train_count:
            os.replace(os.path.join(final_dir, file), os.path.join(final_dir,'train',file))
            num+=1
            print(f'File number {num} moved to training folder')
        elif num < train_count + test_count:
            os.replace(os.path.join(final_dir, file), os.path.join(final_dir,'test',file))
            num+=1
            print(f'File number {num} moved to testing folder')
        elif num < train_count + test_count + val_count:
            os.replace(os.path.join(final_dir, file), os.path.join(final_dir,'val',file))
            num+=1
            print(f'File number {num} moved to validation folder')

def dicom_to_ml():
    dicom_dir = askdirectory(title='Select DICOM Directory')
    train_split = askstring(title="Training Folder", prompt="Input train split (between 0 and 1):")
    test_split = askstring(title="Testing Folder", prompt="Input test split (between 0 and 1):")
    val_split = askstring(title="Validation Folder", prompt="Input validation split (between 0 and 1):")
    dicom_path_to_ml_model(dicom_dir, train_split, test_split, val_split)

def read_nii():
    label = nib.load('image123.nii')
    image = label.get_data()
    plt.imshow(image)
    plt.show()

    
def create_jpeg_dataset():

    num = 0

    for finding in FINDINGS:
        print(finding)
        dir_path = f'{desktop_path}med_img_ai/ml_jpegs/{finding}'
        finding_num = FINDINGS.index(finding)
        if finding == 'No_finding':
            for file in os.listdir(dir_path):
                img = cv2.imread(f'{desktop_path}med_img_ai/ml_jpegs/{finding}/{file}', 0)
                new_img = cv2.resize(img, (800,800))
                new_img = new_img / 255
                num += 1
                if num < 300:
                    data.append([new_img, finding_num])
                else:
                    break
        else:
            for file in os.listdir(dir_path):
                img = cv2.imread(f'{desktop_path}med_img_ai/ml_jpegs/{finding}/{file}', 0)
                new_img = cv2.resize(img, (800,800))
                new_img = new_img / 255
                data.append([new_img, finding_num])

    random.shuffle(data)
    write_file = open(f'{desktop_path}med_img_ai/storedarr.txt', 'wb')
    pickle.dump(data, write_file)
    write_file.close()

def create_cnn():

    size = 224
    color_channels = 1
    color_mode = 'grayscale'
    batch_size = 32

    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    train_image_generator = image_data_generator.flow_from_directory('chest-xray-imgs/train/model', seed=123, color_mode=color_mode, batch_size=batch_size,class_mode='binary', target_size=(size,size))
    test_image_generator = image_data_generator.flow_from_directory('chest-xray-imgs/test/model', seed=123, color_mode=color_mode, batch_size=batch_size,class_mode='binary', target_size=(size,size))
    val_image_generator = image_data_generator.flow_from_directory('chest-xray-imgs/val/model', seed=123, color_mode=color_mode, batch_size=batch_size,class_mode='binary', target_size=(size,size))

    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=10, verbose=1)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='cxray-nf-mass.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (size,size,color_channels), padding='same'))
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (size,size,color_channels), padding='same'))  
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (size,size,color_channels), padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (size,size,color_channels), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (size,size,color_channels), padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (size,size,color_channels), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (size,size,color_channels), padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (size,size,color_channels), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (size,size,color_channels), padding='same'))
    # model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (size,size,color_channels), padding='same'))
    # model.add(tf.keras.layers.MaxPooling2D((2,2)))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (size,size,color_channels), padding='same'))
    # model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (size,size,color_channels), padding='same'))
    # model.add(tf.keras.layers.MaxPooling2D((2,2)))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (size,size,color_channels), padding='same'))
    # model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (size,size,color_channels), padding='same'))
    # model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    # model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    # model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

    model.compile(optimizer="nadam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(train_image_generator, epochs = 12, validation_data=val_image_generator, callbacks=[es, model_checkpoint])

    model.evaluate(test_image_generator)

def load_model():

    size = 224
    color_mode = 'grayscale'
    batch_size = 32
    
    image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    test_image_generator = image_data_generator.flow_from_directory('chest-xray-imgs/test/model', seed=123, color_mode=color_mode, batch_size=batch_size,class_mode='binary', target_size=(size,size))

    x = test_image_generator.next()

    for i in range(batch_size):
        model = tf.keras.models.load_model('cxray-nf-mass376p.h5')
        ans = model.predict(x[0])
        print(ans[i])
        print(x[1][i])
        plt.imshow(x[0][i])
        plt.show()

dicom_to_ml()
