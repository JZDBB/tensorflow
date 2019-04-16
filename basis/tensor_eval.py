# tensorflow有两种方式：Session.run和 Tensor.eval
# 区别：
#   使用t.eval()时，等价于：tf.get_default_session().run(t)
#   可以使用sess.run()在同一步获取多个tensor中的值，
#   每次使用 eval 和 run时，都会执行整个计算图，为了获取计算的结果，将它分配给tf.Variable，然后获取。

# !!! Session.run（）常用于获取多个tensor中的值，而Tensor.eval()常用于单元测试、获取单个Tensor值 !!!

import tensorflow as tf
t = tf.constant(42.0)
sess = tf.Session()
with sess.as_default():   # or `with sess:` to close on exit
    assert sess is tf.get_default_session()
    assert t.eval() == sess.run(t)


t = tf.constant(42.0)
u = tf.constant(37.0)
tu = tf.multiply(t, u)
ut = tf.multiply(u, t)
with sess.as_default():
   print(tu.eval())  # runs one step
   print(ut.eval())  # runs one step
   print(sess.run([tu, ut]))  # evaluates both tensors in a single step

# tensorflow Version Error
# tf.mul - --tf.multiply
# tf.sub - --tf.subtract
# tf.neg - --tf.negative