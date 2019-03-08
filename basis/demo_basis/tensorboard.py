import tensorflow as tf
import os
import shutil
import numpy as np

def board():
    """
    TensorBoard 简单例子。
    tf.summary.scalar('var_name', var)        # 记录标量的变化
    tf.summary.histogram('vec_name', vec)     # 记录向量或者矩阵，tensor的数值分布变化。
    tf.summary.distribution()                 # 分布图，一般用于显示weights分布
    merged = tf.summary.merge_all()           # 把所有的记录并把他们写到 log_dir 中
    train_writer = tf.summary.FileWriter(log_dir + '/add_example', sess.graph)  # 保存位置
    """
    config  = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    log_dir = 'summary/graph/'
    if os.path.exists(log_dir):   # 删掉以前的summary，以免重合
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    print('created log_dir path')

    with tf.name_scope('add_example'):
        a = tf.Variable(tf.truncated_normal([100,1], mean=0.5, stddev=0.5), name='var_a')
        tf.summary.histogram('a_hist', a)
        b = tf.Variable(tf.truncated_normal([100,1], mean=-0.5, stddev=1.0), name='var_b')
        tf.summary.histogram('b_hist', b)
        increase_b = tf.assign(b, b + 0.2)
        c = tf.add(a, b)
        tf.summary.histogram('c_hist', c)
        c_mean = tf.reduce_mean(c)
        tf.summary.scalar('c_mean', c_mean)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_dir+'add_example', sess.graph)
    sess.run(tf.global_variables_initializer())
    for step in range(500):
        sess.run([merged, increase_b])    # 每步改变一次 b 的值
        summary = sess.run(merged)
        writer.add_summary(summary, step)
    writer.close()

def board2():
    """TensorBoard ——train\test分开
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)  # 保存位置
    test_writer = tf.summary.FileWriter(log_dir + '/test', sess.graph)
    运行完后，在命令行中输入 tensorboard --logdir=log_dir_path(你保存到log路径)
    """
    config  = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    log_dir = 'summary/graph2/'
    if os.path.exists(log_dir):   # 删掉以前的summary，以免重合
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    print('created log_dir path')

    a = tf.placeholder(dtype=tf.float32, shape=[100,1], name='a')

    with tf.name_scope('add_example'):
        b = tf.Variable(tf.truncated_normal([100,1], mean=-0.5, stddev=1.0), name='var_b')
        tf.summary.histogram('b_hist', b)
        increase_b = tf.assign(b, b + 0.2)
        c = tf.add(a, b)
        tf.summary.histogram('c_hist', c)
        c_mean = tf.reduce_mean(c)
        tf.summary.scalar('c_mean', c_mean)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)  # 保存位置
    test_writer = tf.summary.FileWriter(log_dir + '/test', sess.graph)

    sess.run(tf.global_variables_initializer())
    for step in range(500):
        if (step+1) % 10 == 0:
            _a = np.random.randn(100,1)
            summary, _ = sess.run([merged, increase_b], feed_dict={a: _a})    # 每步改变一次 b 的值
            test_writer.add_summary(summary, step)
        else:
            _a = np.random.randn(100,1) + step*0.2
            summary, _ = sess.run([merged, increase_b], feed_dict={a: _a})    # 每步改变一次 b 的值
            train_writer.add_summary(summary, step)
    train_writer.close()
    test_writer.close()

def board3():
    # 迭代的计数器
    global_step = tf.Variable(0, trainable=False)
    # 迭代的+1操作
    increment_op = tf.assign_add(global_step, tf.constant(1))
    # 实例应用中，+1操作往往在`tf.train.Optimizer.apply_gradients`内部完成。

    # 创建一个根据计数器衰减的Tensor
    lr = tf.train.exponential_decay(0.1, global_step, decay_steps=1, decay_rate=0.9, staircase=False)

    # 把Tensor添加到观测中
    tf.summary.scalar('learning_rate', lr)

    # 并获取所有监测的操作`sum_opts`
    sum_ops = tf.summary.merge_all()

    # 初始化sess
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)  # 在这里global_step被赋初值

        # 指定监测结果输出目录
        summary_writer = tf.summary.FileWriter('./logs/', sess.graph)

        # 启动迭代
        for step in range(0, 10):
            s_val = sess.run(sum_ops)  # 获取serialized监测结果：bytes类型的字符串
            summary_writer.add_summary(s_val, global_step=step)  # 写入文件
            sess.run(increment_op)  # 计数器+1
        summary_writer.close()

if __name__ == '__main__':
    # board()
    # board2()
    board3()