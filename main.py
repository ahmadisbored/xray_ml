import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import os

FINDINGS = ['Disc_space_narrowing', 'Foraminal_stenosis', 'No_finding', 'Osteophytes', 'Other_lesions', 'Spondylolysthesis', 'Surgical_implant', 'Vertebral_collapse']

data = []

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
            # else:
            #     while(num < 5):
            #         try:
            #             dicom_bytes = tf.io.read_file(path + f'/{dicom}')
            #             dicom = tfio.image.decode_dicom_image(dicom_bytes, dtype = tf.uint16)
            #             processed_dicom = np.squeeze(dicom.numpy())
            #             processed_dicom = processed_dicom / 65536
            #             data.append([processed_dicom, finding_ind])
            #             num+=1
            #         except:
            #             pass

    write_file = open('C:/Users/almaa/Desktop/med_img_ai/storedarr.txt', 'wb')
    pickle.dump(data, write_file)
    write_file.close()

# create_dicom_dataset()

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

num = random.randint(0,55)
read_plot_dataset(num)


