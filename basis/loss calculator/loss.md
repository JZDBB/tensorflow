# Loss Calculate

## Cross-entropy

### 1、tf.nn.sigmoid_cross_entropy_with_logits

```python
tf.nn.sigmoid_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    name=None
)
```

- Sigmoid其实是Softmax的一个特例，Softmax用于多分类问题，而Sigmoid用于二分类问题。

- 计算公式（其中$x=logits,\ z=labels​$）：

  $Loss=z\times-log(sigmoid(x))+(1-z) \times-log(1-sigmoid(x))​$

### 2、tf.nn.weighted_cross_entropy_with_logits

```python
tf.nn.weighted_cross_entropy_with_logits(
    targets,
    logits,
    pos_weight,
    name=None
)
```

- 多出了一个参数pos_weight，该函数计算带有权重的Sigmoid交叉熵。

- 计算公式（其中$x=logits,\ z=labels$）：

  $Loss=z\times-log(sigmoid(x))\times pos\_weight+(1-z) \times-log(1-sigmoid(x))​$

- **当训练数据不平衡（imbalanced）的情况比较严重时，该损失函数是一个不错的选择。**

### 3、tf.nn.softmax_cross_entropy_with_logits

```python
tf.nn.softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    dim=-1,
    name=None
)
```

- 该函数用于多项互斥分类任务。例如CIFAR-10中图片只能分一类。
- 注意：
  1. 传入的labels必须是已经事先使用**one-hot**编码后的数据，或者是和为1的概率分布。
  2. 传入的logits是**unscaled**的，因为函数内部会更加高效的计算softmax，因此**不要将已经softmax过的数据作为logits传入该函数**。
  3. 该函数在反向传播过程中只会作用于logits，不会同时作用于labels。

### 4、tf.nn.softmax_cross_entropy_with_logits_v2

```python
tf.nn.softmax_cross_entropy_with_logits_v2(
    _sentinel=None,
    labels=None,
    logits=None,
    dim=-1,
    name=None
)
```

和3中函数唯一的差别在于反向传播时会同时作用于logits和labels。如果不想让labels参与到反向传播中，请在调用该函数前，将label的tensor传递给`tf.stop_gradient`。

### 5、tf.nn.sparse_softmax_cross_entropy_with_logits

```python
tf.nn.sparse_softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    name=None
)
```

和3中函数差别在于labels的编码方式不同。该函数的labels不是one-hot向量，而是每个类别的**index**。考虑一种情况，当类别数量很大时，如果使用one-hot向量表示每一个类别的话，会占用很大的空间，造成不必要的浪费，这时用类别的index表示会一定程度上缓解这一个问题。

### 6、tf.nn.sampled_sotfmax_loss

```python
tf.nn.sampled_softmax_loss(
    weights,
    biases,
    labels,
    inputs,
    num_sampled,
    num_classes,
    num_true=1,
    sampled_values=None,
    remove_accidental_hits=True,
    partition_strategy='mod',
    name='sampled_softmax_loss',
    seed=None
)
```

当训练数据的类别很多时，计算准确的概率分布会耗费很大的计算资源，该函数就是通过计算一个随机样本的loss值来估计整体的概率分布。而且该函数仅在训练的时候才会被调用。如下这个例子：

```python
if mode == "train":
  loss = tf.nn.sampled_softmax_loss(
      weights=weights,
      biases=biases,
      labels=labels,
      inputs=inputs,
      ...,
      partition_strategy="div")
elif mode == "eval":
  logits = tf.matmul(inputs, tf.transpose(weights))
  logits = tf.nn.bias_add(logits, biases)
  labels_one_hot = tf.one_hot(labels, n_classes)
  loss = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels_one_hot,
      logits=logits)
```

