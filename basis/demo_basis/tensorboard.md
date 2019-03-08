# Tensorboard

## summary
`Summary`是对网络中`Tensor`取值进行监测的一种`Operation`, 是“外围”操作，不影响数据流本身。

**注意：Summary本身也是一个op，op是需要sess.run的**

#### Tensorboard数据形式

（1）标量Scalars： accuracy，cross entropy，dropout，layer1 和 layer2 的 bias 和 weights 等的趋势
（2）图片Images：输入的数据
（3）音频Audio：输入的数据
（4）计算图Graph：模型的结构
（5）数据分布Distribution：activations，gradients 或者 weights 等变量的每一步的分布
（6）直方图Histograms： activations，gradients 或者 weights 等变量的每一步的分布
（7）嵌入向量Embeddings： PCA 主成分分析方法将高维数据投影到 3D 空间后的数据的关系

#### Tensorboard可视化

1. 建立一个graph
2. 确定要在graph中的哪些节点放置summary operations以记录信息 
   使用tf.summary.scalar记录标量 
   使用tf.summary.histogram记录数据的直方图 
   使用tf.summary.distribution记录数据的分布图 
   使用tf.summary.image记录图像数据 
   ….
3. sess.run(op) 或者sess.run(op->依赖之)
4. 使用**tf.summary.FileWriter**将运行后输出的数据都保存到本地磁盘中
5. 运行整个程序，并在命令行输入运行tensorboard的指令，之后打开web端可查看可视化的结果

#### Example

```python
# 迭代的计数器
global_step = tf.Variable(0, trainable=False)
# 迭代的+1操作
increment_op = tf.assign_add(global_step, tf.constant(1))
# 实例应用中，+1操作往往在`tf.train.Optimizer.apply_gradients`内部完成。

# 创建一个根据计数器衰减的Tensor
lr = tf.train.exponential_decay(0.1, global_step, decay_steps=1, decay_rate=0.9, staircase=False)

# 把Tensor添加到观测中
tf.scalar_summary('learning_rate', lr)

# 并获取所有监测的操作`sum_opts`
sum_ops = tf.merge_all_summaries()

# 初始化sess
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)  # 在这里global_step被赋初值

# 指定监测结果输出目录
summary_writer = tf.train.SummaryWriter('/tmp/log/', sess.graph)

# 启动迭代
for step in range(0, 10):
    s_val = sess.run(sum_ops)    # 获取serialized监测结果：bytes类型的字符串
    summary_writer.add_summary(s_val, global_step=step)   # 写入文件
    sess.run(increment_op)     # 计数器+1
```

