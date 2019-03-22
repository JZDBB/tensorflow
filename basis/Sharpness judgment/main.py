import tensorflow as tf
import numpy as np
import read_tfrecords
import net
import os
import input

flags = tf.app.flags

flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
flags.DEFINE_string('data_dir', '/home/yqi/Desktop/yn/tensorflow/basis/Sharpness judgment/train.tfrecords', 'data direction')
flags.DEFINE_string('log_dir', './logs', 'log direction')
flags.DEFINE_string('ckpt_dir', './models', 'check point direction')
flags.DEFINE_float('weight_decay', 0.0001, 'weight decay')
flags.DEFINE_integer('decay_steps', 100, 'decay steps')
flags.DEFINE_float('decay_rate', 0.95, 'decay rate')
flags.DEFINE_float('momentum', 0.9, 'momentum')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('patch_size', 32, '')
flags.DEFINE_float('dropout', 0.5, 'keep probability')
flags.DEFINE_integer('max_steps', 6400, 'max steps')
flags.DEFINE_integer('start_step', 1, 'start steps')

FLAGS = flags.FLAGS


def main(_):

    graph = tf.Graph()
    with graph.as_default():

        # img, label = read_tfrecords.read_and_decode("train.tfrecords")
        train_example_batch, train_label_batch = input.input_pipeline(
            tf.train.match_filenames_once(FLAGS.data_dir), FLAGS.batch_size,
            FLAGS.patch_size)

        valid_example_batch, valid_label_batch = input.input_pipeline(
            tf.train.match_filenames_once(FLAGS.data_dir), FLAGS.batch_size,
            FLAGS.patch_size)

        global_step = tf.Variable(FLAGS.start_step, name="global_step")
        learning_rate = tf.train.piecewise_constant(global_step, [3200, 4800], [0.1, 0.01, 0.001])

        image_place = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='image')
        label_place = tf.placeholder(tf.float32, shape=(None, 1), name='labels')
        dropout_param = tf.placeholder(tf.float32)

        # img_batch, label_batch = tf.train.shuffle_batch([img, label],
        #                                                 batch_size=FLAGS.batch_size, capacity=2000,
        #                                                 min_after_dequeue=1000)

        Net = net.resnet(20)
        feat = Net._build_net(image_place, Net.n)
        pred = tf.contrib.layers.fully_connected(feat, 1, None,
                                                 weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                     FLAGS.weight_decay))
        loss = tf.reduce_sum(tf.multiply(tf.square(pred - label_place), 1.0/FLAGS.batch_size))
        train_step = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum).minimize(
            loss, global_step=global_step)
        tf.summary.scalar("loss", loss, collections=['train', 'test']) #
        tf.summary.scalar("learning_rate", learning_rate, collections=['train'])
        train = tf.summary.merge_all()


        with tf.Session() as sess:
            saver = tf.train.Saver(name="saver")
            if tf.gfile.Exists(os.path.join(FLAGS.ckpt_dir, 'checkpoint')):
                saver.restore(sess, os.path.join(FLAGS.ckpt_dir, 'model.ckpt'))
            else:
                sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(FLAGS.log_dir + '/logs', sess.graph)
            writer.flush()
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            # 使用start_queue_runners 启动队列填充
            threads = tf.train.start_queue_runners(sess, coord)
            def feed_dict(train, on_training):
                def get_batch(data, labels):
                    d, l = sess.run([data, labels])
                    d = d.astype(np.float32)
                    l = l.astype(np.float32)
                    return d, l

                if train:
                    xs, ys = get_batch(train_example_batch, train_label_batch)
                else:
                    xs, ys = get_batch(valid_example_batch, valid_label_batch)
                    pass
                return {image_place: xs, label_place: ys}

            for epoch in range(FLAGS.max_steps):

                for i in range(2000):
                    batch_loss, _ = sess.run([loss, train_step],
                                feed_dict=feed_dict(True, True))

                            # writer.add_summary(train_summaries, global_step=i)
                    print('Loss, '+"{:.3f}".format(batch_loss))

                print("Epoch #" + str(epoch + 1) + ", Train Loss=" + \
                       "{:.3f}".format(batch_loss))
                if epoch % 20 == 0:

                    save_path = saver.save(sess, FLAGS.ckpt_dir)
                    print("Model saved in file: %s" % save_path)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()