import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# pathes = os.listdir('./img/')
# num = 0
# data_lists = []
# for path in pathes:
#     num += 1
#     print(path)
#     for i in range(1, 10):
#         img = cv2.imread('./img/%s' % path)
#         h, w, c = img.shape
#         img = cv2.resize(img, (int(w*i/10), int(h*i/10)))
#         img = cv2.resize(img, (w, h))
#         cv2.imwrite('./data/%s_%s.jpg' % (num, i), img)
#         # 图片另存到output文件夹中，图片质量压缩到60%
#         # cv2.imwrite('./data/test%s%s.jpg' % (num, i), img, [int(cv2.IMWRITE_JPEG_QUALITY), i*10])
#         # img = cv2.imread('./data/test%s%s.jpg' % (num, i))
#         # arr = np.array(img)
#         # data_lists.append([arr, i^2])
#     img = cv2.imread('./img/%s' % path)
#     cv2.imwrite('./data/%s_%s.jpg' % (num, 0), img)


# path = "./data/"
# def data_load(path):
#     filenames = os.listdir(path)
#     for filename in filenames:
#         str = filename.split('.')[0]
#         label = str.split('_')[0]
#         img = plt.imread(filename)
#         im_arr = np.array(img)

def remove(list_target, list_sample):
    res = []
    for item in list_target:
        if item in list_sample:
            continue
        else:
            res.append(item)
    return res

x = [m for m in range(1, 257)]
len = len(x)
val_list = random.sample(x, int(len/8))
res = remove(x, val_list)
test_list = random.sample(res, int(len/8))
train_list = remove(res, test_list)

train = []
test = []
val = []
for i in range(1, 257):
    for j in range(0, 10):
        if j == 0:
            label = 100
        else:
            label = pow(j,2)
        if i in train_list:
            train.append(['%s_%s.jpg' % (i, j), label])
        if i in test_list:
            test.append(['%s_%s.jpg' % (i, j), label])
        if i in val_list:
            val.append(['%s_%s.jpg' % (i, j), label])

writer= tf.python_io.TFRecordWriter("val.tfrecords")
for data in val:
    for i in range(200):
        img = plt.imread(os.path.join('./data', data[0]))
        x = random.randint(0, img.shape[0] - 128)
        y = random.randint(0, img.shape[1] - 128)
        img = img[x:x + 128, y:y + 128, :]
        patch_raw = img.tostring()
        label_raw = np.array([data[1]]).astype(np.float32).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'patch_raw': _bytes_feature(patch_raw),
            'label': _bytes_feature(label_raw)}))
        writer.write(example.SerializeToString())
    print(data[0])
writer.close()

writer= tf.python_io.TFRecordWriter("test.tfrecords")
for data in test:
    for i in range(200):
        img = plt.imread(os.path.join('./data', data[0]))
        x = random.randint(0, img.shape[0] - 128)
        y = random.randint(0, img.shape[1] - 128)
        img = img[x:x + 128, y:y + 128, :]
        patch_raw = img.tostring()
        label_raw = np.array([data[1]]).astype(np.float32).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'patch_raw': _bytes_feature(patch_raw),
            'label': _bytes_feature(label_raw)}))
        writer.write(example.SerializeToString())
    print(data[0])
writer.close()


writer= tf.python_io.TFRecordWriter("train.tfrecords")
for data in train:
    for i in range(200):
        img = plt.imread(os.path.join('./data', data[0]))
        x = random.randint(0, img.shape[0] - 128)
        y = random.randint(0, img.shape[1] - 128)
        img = img[x:x + 128, y:y + 128, :]
        patch_raw = img.tostring()
        label_raw = np.array([data[1]]).astype(np.float32).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'patch_raw': _bytes_feature(patch_raw),
            'label': _bytes_feature(label_raw)}))
        writer.write(example.SerializeToString())
    print(data[0])
writer.close()