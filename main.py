import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import pydicom
from PIL import Image
import cv2
import os

FINDINGS = []

data = []


def dicom_to_jpeg(file_path, file_name,finding):
    ds = pydicom.dcmread(file_path)
    new_image = ds.pixel_array.astype(float)
    scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
    scaled_image = np.uint8(scaled_image)
    final_image = Image.fromarray(scaled_image)
    try:
        final_image.save(f'C:/Users/almaa/Desktop/med_img_ai/ml_jpegs/{finding}/{file_name}.jpg')
    except:
        os.mkdir(f'C:/Users/almaa/Desktop/med_img_ai/ml_jpegs/{finding}')
        final_image.save(f'C:/Users/almaa/Desktop/med_img_ai/ml_jpegs/{finding}/{file_name}.jpg')

def dicom_to_jpeg_sort():
    for finding in FINDINGS:
        dir_path = f'C:/Users/almaa/Desktop/med_img_ai/ml_imgs/{finding}'
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                print(file)
                file_path = f'C:/Users/almaa/Desktop/med_img_ai/ml_imgs/{finding}/{file}'
                file_name = str(file[:-6])
                dicom_to_jpeg(file_path, file_name, finding)

def read_plot_dataset(num):
    read_file = open('C:/Users/almaa/Desktop/med_img_ai/storedarr.txt', 'rb')
    try:
        dataset = pickle.load(read_file)
    except:
        pass
    read_file.close()
    plt.imshow(dataset[num][0], cmap='gray')
    plt.title(FINDINGS[dataset[num][1]])
    plt.show()

def create_jpeg_dataset():

    num = 0

    for finding in FINDINGS:
        print(finding)
        dir_path = f'C:/Users/almaa/Desktop/med_img_ai/ml_jpegs/{finding}'
        finding_num = FINDINGS.index(finding)
        if finding == 'No_finding':
            for file in os.listdir(dir_path):
                img = cv2.imread(f'C:/Users/almaa/Desktop/med_img_ai/ml_jpegs/{finding}/{file}', 0)
                new_img = cv2.resize(img, (800,800))
                new_img = new_img / 255
                num += 1
                if num < 300:
                    data.append([new_img, finding_num])
                else:
                    break
        else:
            for file in os.listdir(dir_path):
                img = cv2.imread(f'C:/Users/almaa/Desktop/med_img_ai/ml_jpegs/{finding}/{file}', 0)
                new_img = cv2.resize(img, (800,800))
                new_img = new_img / 255
                data.append([new_img, finding_num])

    random.shuffle(data)
    write_file = open('C:/Users/almaa/Desktop/med_img_ai/storedarr.txt', 'wb')
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

load_model()
