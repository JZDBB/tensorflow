import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import net
import os

flags = tf.app.flags

flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
flags.DEFINE_string('data_dir', 'train_data.pickle', 'data direction')
flags.DEFINE_string('log_dir', './logs', 'log direction')
flags.DEFINE_string('ckpt_dir', './models', 'check point direction')
flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
flags.DEFINE_integer('decay_steps', 100, 'decay steps')
flags.DEFINE_float('decay_rate', 0.95, 'decay rate')
flags.DEFINE_float('momentum', 0.9, 'momentum')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_float('dropout', 0.5, 'keep probability')
flags.DEFINE_integer('max_steps', 6400, 'max steps')
flags.DEFINE_integer('start_step', 1, 'start steps')

FLAGS = flags.FLAGS


def load_data_pickle(filenames):
    data = []
    labels = []
    with open(filenames, 'rb') as f:
        datasets = pickle.load(f)
        random.shuffle(datasets)
        for dataset in datasets:
            img = dataset[0]
            x = random.randint(0, img.shape[0]-32)
            y = random.randint(0, img.shape[1]-32)
            img = img[x:x+32, y:y+32, :]
            img_arr = np.array(img)
            label = dataset[1]
            data.append(img_arr)
            labels.append(label)
    return [data, labels]

def load_data(filenames):
    data = []
    labels = []
    with open(filenames, 'rb') as f:
        datasets = pickle.load(f)
        random.shuffle(datasets)
        for dataset in datasets:
            data.append(dataset[0])
            labels.append(dataset[1])
    return [data, labels]

def main(_):

    # data = load_data(FLAGS.data_dir)
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.Variable(FLAGS.start_step, name="global_step")
        learning_rate = tf.train.piecewise_constant(global_step, [3200, 4800], [0.1, 0.01, 0.001])

        image_place = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='image')
        label_place = tf.placeholder(tf.float32, shape=(None, 1), name='labels')
        dropout_param = tf.placeholder(tf.float32)

        Net = net.resnet(20)
        feat = Net._build_net(image_place, Net.n)
        pred = tf.contrib.layers.fully_connected(feat, 1, None,
                                                 weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                     FLAGS.weight_decay))
        loss = tf.reduce_sum(tf.multiply(tf.square(pred - label_place), 1.0/FLAGS.batch_size))
        train_step = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum).minimize(
            loss, global_step=global_step)
        tf.summary.scalar("loss", loss, collections=['train', 'test'])
        tf.summary.scalar("learning_rate", learning_rate, collections=['train'])
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            saver = tf.train.Saver(name="saver")
            if tf.gfile.Exists(os.path.join(FLAGS.ckpt_dir, 'checkpoint')):
                saver.restore(sess, os.path.join(FLAGS.ckpt_dir, 'model.ckpt'))
            else:
                sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter(FLAGS.log_dir + '/logs', sess.graph)
            writer.flush()

            for epoch in range(FLAGS.max_steps):
                data = load_data(FLAGS.data_dir)
                image = np.array(data[0])
                labels = np.reshape(np.array(data[1]), (-1, 1))
                # print(labels.shape)
                for i in range(int(len(data[0])/FLAGS.batch_size)):
                    start_idx = i * FLAGS.batch_size
                    end_idx = (i + 1) * FLAGS.batch_size

                    train_batch_data, train_batch_label = image[start_idx:end_idx], labels[start_idx:end_idx]
                    batch_loss, _ = sess.run([loss, train_step],
                        feed_dict={image_place: train_batch_data,
                                   label_place: train_batch_label})

                    # writer.add_summary(train_summaries, global_step=i)
                    # print('Loss, '+"{:.3f}".format(batch_loss))

                print("Epoch #" + str(epoch + 1) + ", Train Loss=" + \
                       "{:.3f}".format(batch_loss))
                if epoch % 20 == 0:

                    save_path = saver.save(sess, FLAGS.ckpt_dir)
                    print("Model saved in file: %s" % save_path)




if __name__ == '__main__':
    tf.app.run()