import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


classes={'iris','contact'}
writer= tf.python_io.TFRecordWriter("iris_contact.tfrecords")

for index,name in enumerate(classes):
    class_path='/'
    for img_name in os.listdir(class_path):
        img_path=class_path+img_name
        img=Image.open(img_path)
        img= img.resize((512,80))
        img_raw=img.tobytes()
        #plt.imshow(img) # if you want to check you image,please delete '#'
        #plt.show()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())

writer.close()