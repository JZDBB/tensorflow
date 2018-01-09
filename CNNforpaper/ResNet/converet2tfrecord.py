# TFrecords结构
# message Example {
#  Features features = 1;
# };
# message Features{
#  map<string,Feature> featrue = 1;
# };
# message Feature{
#     oneof kind{
#         BytesList bytes_list = 1;
#         FloatList float_list = 2;
#         Int64List int64_list = 3;
#     }};



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

Train_size = 50000
Test_size = 10000

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

writer= tf.python_io.TFRecordWriter("traindata.tfrecords")
for i in range(Train_size):
    img= train_data[i]
    # img= img.resize((32, 32))
    img_raw=img.tobytes()
    # plt.imshow(img)
    # plt.show()
    example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[train_labels[i]])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    #主要的语句
    # tf.train.Feature(int64_list=tf.train.Int64List(value=[int_scalar]))
    # tf.train.Feature(bytes_list=tf.train.BytesList(value=[array_string_or_byte]))
    # tf.train.Feature(bytes_list=tf.train.FloatList(value=[float_scalar]))

    writer.write(example.SerializeToString())

writer.close()

writer= tf.python_io.TFRecordWriter("testdata.tfrecords")
for i in range(Test_size):
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