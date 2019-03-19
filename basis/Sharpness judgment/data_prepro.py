import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

# pathes = os.listdir('./img/')
# num = 0
# data_lists = []
# for path in pathes:
#     num += 1
#     print(path)
#     # cv2.imwrite('./data/%s.jpg' % num, img)
#
#     for i in range(1, 10):
#         img = cv2.imread('./img/%s' % path)
#         h, w, c = img.shape
#         img = cv2.resize(img, (int(w*i/10), int(h*i/10)))
#         img = cv2.resize(img, (w, h))
#         cv2.imwrite('./data/%s_%s.jpg' % (num, i), img)
        # 图片另存到output文件夹中，图片质量压缩到60%
        # cv2.imwrite('./data/test%s%s.jpg' % (num, i), img, [int(cv2.IMWRITE_JPEG_QUALITY), i*10])
        # img = cv2.imread('./data/test%s%s.jpg' % (num, i))
#         arr = np.array(img)
#         data_lists.append([arr, i^2])
#
# pickle.dump(data_lists, 'data.pickle')


# path = "./data/"
# def data_load(path):
#     filenames = os.listdir(path)
#     for filename in filenames:
#         str = filename.split('.')[0]
#         label = str.split('_')[0]
#         img = plt.imread(filename)
#         im_arr = np.array(img)

data_lists=[]
for i in range(150):
    for j in range(1, 10):
        data_lists.append(['%s_%s.jpg' % (i, j), j^2])


def remove(list_target, list_sample):
    for item in list_sample:
        if item in list_target:
            list_target.remove(item)
    return list_target

# train_list = []
# val_list = []
# test_list = []
len = len(data_lists)
val_list = random.sample(data_lists, int(len/9))
data_lists = remove(data_lists, val_list)
test_list = random.sample(data_lists, int(len/9))
data_lists = remove(data_lists, test_list)
with open('val.pickle', 'wb') as f:
    pickle.dump(val_list, f)

with open('test.pickle', 'wb') as f:
    pickle.dump(test_list, f)

with open('train.pickle', 'wb') as f:
    pickle.dump(data_lists, f)

