#使用队列读取TFrecords

#其中需要注意的事情！！
# 第一，tensorflow里的graph能够记住状态（state），这使得TFRecordReader能够记住tfrecord的位置，
# 并且始终能返回下一个。而这就要求我们在使用之前，必须初始化整个graph，这里我们使用了
# 函数tf.initialize_all_variables()来进行初始化。
#
# 第二，tensorflow中的队列和普通的队列差不多，不过它里面的operation和tensor都是符号型的（symbolic），
# 在调用sess.run()时才执行。
#
# 第三， TFRecordReader会一直弹出队列中文件的名字，直到队列为空。

import tensorflow as tf

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [32, 32, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label

#训练的时候调用example
img, label = read_and_decode("train.tfrecords")

#使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=30, capacity=2000,
                                                min_after_dequeue=1000)
# tf.train.shuffle_batch函数输入参数为：
# tensor_list: 进入队列的张量列表The list of tensors to enqueue.
# batch_size: 从数据队列中抽取一个批次所包含的数据条数The new batch size pulled from the queue.
# capacity: 队列中最大的数据条数An integer. The maximum number of elements in the queue.
# min_after_dequeue: 提出队列后，队列中剩余的最小数据条数Minimum number elements in the queue after a dequeue, used to ensure a level of mixing of elements.
# num_threads: 进行队列操作的线程数目The number of threads enqueuing tensor_list.
# seed: 队列中进行随机排列的随机数发生器，似乎不常用到Seed for the random shuffling within the queue.
# enqueue_many: 张量列表中的每个张量是否是一个单独的例子，似乎不常用到Whether each tensor in tensor_list is a single example.
# shapes: (Optional) The shapes for each example. Defaults to the inferred shapes for tensor_list.
# name: (Optional) A name for the operations.
# 值得注意的是，capacity>=min_after_dequeue+num_threads*batch_size。

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(3):
        val, l= sess.run([img_batch, label_batch])
        #我们也可以根据需要对val， l进行处理
        #l = to_categorical(l, 12)
        print(val.shape, l)