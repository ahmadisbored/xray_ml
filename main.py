import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import pydicom
from PIL import Image
import cv2
import os

FINDINGS = ['Disc_space_narrowing', 'Foraminal_stenosis', 'No_finding', 'Osteophytes', 'Other_lesions', 'Spondylolysthesis', 'Surgical_implant', 'Vertebral_collapse']

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

def create_dicom_dataset():
    for finding in FINDINGS:
        num = 0
        path = f'C:/Users/almaa/Desktop/med_img_ai/ml_imgs/{finding}'
        print(path)
        finding_ind = FINDINGS.index(finding)
        for dicom in os.listdir(path):
            if num < 7:
                try:
                    dicom_bytes = tf.io.read_file(path + f'/{dicom}')
                    dicom = tfio.image.decode_dicom_image(dicom_bytes, dtype = tf.uint16)
                    processed_dicom = np.squeeze(dicom.numpy())
                    processed_dicom = processed_dicom / 65536
                    data.append([processed_dicom, finding_ind])
                    num+=1
                except:
                    pass

    random.shuffle(data)
    write_file = open('C:/Users/almaa/Desktop/med_img_ai/storedarr.txt', 'wb')
    pickle.dump(data, write_file)
    write_file.close()

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

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    read_file = open('C:/Users/almaa/Desktop/med_img_ai/storedarr.txt', 'rb')
    try:
        dataset = pickle.load(read_file)
        print(len(dataset))
    except:
        pass
    
    for i in range(0,1031):
        X_train.append(dataset[i][0])
        Y_train.append(dataset[i][1])
    
    for i in range(1031, 1375):
        X_test.append(dataset[i][0])
        Y_test.append(dataset[i][1])

    X_train = np.array(X_train).reshape(-1,800,800,1)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test).reshape(-1,800,800,1)
    Y_test = np.array(Y_test)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = X_train.shape[1:]))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = X_train.shape[1:]))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = X_train.shape[1:]))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation = 'relu'))
    model.add(tf.keras.layers.Dense(8, activation = 'softmax'))

    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(X_train, Y_train, epochs = 10)

    model.evaluate(X_test, Y_test)

    model.save('preliminary_model.h5')





