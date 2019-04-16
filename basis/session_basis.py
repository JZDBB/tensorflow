# tf.Session()
# 创建一个会话，当上下文管理器退出时会话关闭和资源释放自动完成。
#
# tf.Session().as_default()
# 创建一个默认会话，当上下文管理器退出时会话没有关闭，还可以通过调用会话进行run()和eval()操作.
import tensorflow as tf
a = tf.constant(1.0)
b = tf.constant(2.0)
with tf.Session() as sess:
   print(a.eval())
print(b.eval(session=sess))

a = tf.constant(1.0)
b = tf.constant(2.0)
with tf.Session().as_default() as sess:
   print(a.eval())
   # sess.close() 加上这句话效果就一样啦
print(b.eval(session=sess))