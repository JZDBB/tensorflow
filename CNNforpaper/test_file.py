import numpy as np
import cv2
import matplotlib.pyplot as plt
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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

# dict = unpickle('cifar-10-python\data_batch_1')
# labels = dict[b'labels']
# data = dict[b'data']
plt.imshow(train_data[0])
plt.show()

# train = tuple(train_data)

print(1)