import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
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
            if num < 5:
                try:
                    dicom_bytes = tf.io.read_file(path + f'/{dicom}')
                    dicom = tfio.image.decode_dicom_image(dicom_bytes, dtype = tf.uint16)
                    processed_dicom = np.squeeze(dicom.numpy())
                    processed_dicom = processed_dicom / 65536
                    data.append([processed_dicom, finding_ind])
                    num+=1
                except:
                    pass

    write_file = open('C:/Users/almaa/Desktop/med_img_ai/storedarr.txt', 'wb')
    pickle.dump(data, write_file)
    write_file.close()

create_dicom_dataset()


# image_bytes = tf.io.read_file('first.dicom')
# image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
# processed = np.squeeze(image.numpy())

# plt.imshow(processed, cmap='gray')
# plt.show()
