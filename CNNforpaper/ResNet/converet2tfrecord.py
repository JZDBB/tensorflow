import os
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_data():
    dict_train = unpickle('cifar-10-python\data_batch_1')
    train_data = dict_train[b'data']
    train_labels = dict_train[b'labels']
    dict_train = unpickle('cifar-10-python\data_batch_2')
    train_data = np.append(train_data, dict_train[b'data'], axis=0)
    train_labels.extend(dict_train[b'labels'])
    dict_train = unpickle('cifar-10-python\data_batch_3')
    train_data = np.append(train_data, dict_train[b'data'], axis=0)
    train_labels.extend(dict_train[b'labels'])
    dict_train = unpickle('cifar-10-python\data_batch_4')
    train_data = np.append(train_data, dict_train[b'data'], axis=0)
    train_labels.extend(dict_train[b'labels'])
    dict_train = unpickle('cifar-10-python\data_batch_5')
    train_data = np.append(train_data, dict_train[b'data'], axis=0)
    train_labels.extend(dict_train[b'labels'])
    dict_test = unpickle('cifar-10-python\\test_batch')
    test_data = dict_test[b'data']
    test_labels = dict_test[b'labels']
    train_data = np.reshape(train_data, [-1, 32, 32, 3], 'F')
    train_data = np.transpose(train_data, [0, 2, 1, 3])
    test_data = np.reshape(test_data, [-1, 32, 32, 3], 'F')
    test_data = np.transpose(test_data, [0, 2, 1, 3])
    return train_data, train_labels, test_data, test_labels

classes={'iris','contact'}
train_data, train_labels, test_data, test_labels = read_data()

# writer= tf.python_io.TFRecordWriter("traindata.tfrecords")
# for i in range(50000):
#     img= train_data[i]
#     # img= img.resize((32, 32))
#     img_raw=img.tobytes()
#     # plt.imshow(img)
#     # plt.show()
#     example = tf.train.Example(features=tf.train.Features(feature={
#         "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[train_labels[i]])),
#         'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
#     }))
#     writer.write(example.SerializeToString())
#
# writer.close()

writer= tf.python_io.TFRecordWriter("testdata.tfrecords")
for i in range(10000):
    img= test_data[i]
    # img= img.resize((32, 32))
    img_raw=img.tobytes()
    # plt.imshow(img)
    # plt.show()
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[test_labels[i]])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    writer.write(example.SerializeToString())

writer.close()