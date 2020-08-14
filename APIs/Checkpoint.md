## 引言

很多时候，我们希望在模型训练完成后能将训练好的参数（变量）保存起来，在需要使用模型的其他地方载入模型和参数，就能直接得到训练好的模型。而TensorFlow 正好提供了`tf.train.Checkpoint`这一强大的变量保存与恢复类，可以使用其 `save()` 和 `restore()` 方法将 TensorFlow 中所有包含 Checkpointable State 的对象进行保存和恢复。不过有一点要注意的是，Checkpoint 只保存模型的参数，即Checkpoint中存储着模型model所使用的的所有的`tf.Variable`对象，不保存模型的计算过程，因此一般用于在具有模型源代码的时候恢复之前训练好的模型参数。如果需要导出模型（无需源代码也能运行模型），应该考虑使用SaveModel或者h5格式进行保存。

Checkpoints文件本质是一个二进制文件，它把变量名映射到对应的tensor值 。所以只存储的各个变量的值，并没有网络结构信息！

## Checkpoints的几个文件说明

#### data文件
ckpt-1.data-00000-of-00001：数据文件，保存的是网络的权值，偏置，操作等等。

#### index文件
ckpt-1.index：是一个不可变得字符串字典，每一个键都是张量的名称，它的值是一个序列化的BundleEntryProto。 每个BundleEntryProto描述张量的元数据，所谓的元数据就是描述这个Variable 的一些信息的数据。 “数据”文件中的哪个文件包含张量的内容，该文件的偏移量，校验和，一些辅助数据等等。

#### checkpoint文件——文本文件
checkpoint是一个文本文件，记录了训练过程中在所有中间节点上保存的模型的名称，首行记录的是最后（最近）一次保存的模型名称。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200814235140636.png#pic_center#pic_center)

## checkpoint信息的查看
前面说了，检查点checkpoint的本质是存储的每一个变量的数据，而在index文件中还存储着每一个Variable的名称以及它的元数据，我怎么查看checkpoint中的数据呢？
```python
from tensorflow.python.tools import inspect_checkpoint as chkp
 
# 查看模型中所有的Tensor的数据，这里的默认的all_tensor=True
print(chkp.print_tensors_in_checkpoint_file("./ckpt_model/keypoint_model.ckpt", tensor_name='', all_tensors=True))
 
#print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
''' 输出的格式如下：
tensor_name: tensor1
... ...
tensor_name: tensor2
... ...
tensor_name: tensor3
... ...
'''
```

如果我只想查看一个tensor的信息呢，当然，我需要这个tensor的名称我才能找得到，我可以像下面这样：

```python

# 获取最后保存的一个checkpoint，返回的是最后一个checkpoint的文件路径
model_file = tf.train.latest_checkpoint("./ckpt_model")
print(model_file) # ./ckpt_model/keypoint_model.ckpt-9
 
# 查看其中的某一个张量，此时的all_tensors=False
print(chkp.print_tensors_in_checkpoint_file(model_file, tensor_name="dense4_weights", all_tensors=False))
 
# 当然这里我也可以自己直接传递进去最新的一个checkpoint文件路径
# print(chkp.print_tensors_in_checkpoint_file("./ckpt_model/keypoint_model.ckpt-9" tensor_name="dense4_weights", all_tensors=False))
 
'''
tensor_name:  dense4_weights
[[ 4.58419085e-01  8.78509760e-01  2.11921871e-01 -5.90530671e-02
  -6.23271286e-01 -1.86214507e-01 -3.88072550e-01  1.38646924e+00
   8.91906798e-01  4.05663669e-01]
 ... ...中间是我自己省略了，应该是一个（32,10）的矩阵
 [-1.12723565e+00 -1.26929128e+00 -2.32065111e-01 -6.23432040e-01
  -3.33134890e-01 -9.74284112e-01 -6.22953475e-02 -5.75510025e-01
  -8.32203925e-01  1.12205319e-01]]
'''
```

## 实例
Checkpoint的构造器接受关键字参数，其值是包含可跟踪状态的类型，如`tf.keras.optimizers.Optimizer`的实现中， `tf.Variable S`， `tf.data.Dataset`迭代器， `tf.keras.Layer`的实现，或`tf.keras.Model`的实现。它将这些值保存为checkpoint，并维护一个`save_counter`对checkpoints进行编号。

```python
import tensorflow as tf
import os

checkpoint_directory = "/tmp/training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

#创建一个Checkpoint用来管理两个具有可跟踪状态的对象，两个对象分别命名为“optimizer”和“model”
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
for _ in range(num_training_steps):
	optimizer.minimize( ... )  # 在创建过程中Variables将会被保存
status.assert_consumed()  # 完整性检查(可选)
checkpoint.save(file_prefix=checkpoint_prefix)

```

`Checkpoint.save()`和`Checkpoint.restore()`写入和读取基于对象的checkpoints，而相比之下，TensorFlow 1.x中的`tf.compat.v1.train.Saver`写入和读取基于checkpoints的`variable.name`。基于对象的checkpoints保存Python对象（`Layer`， `Optimizer` ， `Variable`等）之间的依赖关系，该图恢复的checkpoints时，用于匹配变量，所以它在Python程序中拥有更好的健壮性，并支持在恢复基础上创建变量。

`Checkpoint`对象持有作为关键字参数传递给它们的构造函数的对象依赖关系，同时每个依赖会被赋予一个名称，该名称依据传入的关键字参数的名称而创建的。 在TensorFlow类中，向`Layer`和`Optimizer`这样的类会自动为它们的变量添加依赖（例如`tf.keras.layers.Dense` 的"kernel" 和"bias"）。用户自定义的类继承`tf.keras.Model`，更加易于管理依赖，因为`Model`有分配属性的钩子。想下面这样：

```python
class Regress(tf.keras.Model):

  def __init__(self):
    super(Regress, self).__init__()
    self.input_transform = tf.keras.layers.Dense(10)
    # ...

  def call(self, inputs):
    x = self.input_transform(inputs)
    # ...
```

上面这个`Model`有一个名为“input_transform”的Dense层依赖。这使得`Regress`使用`tf.train.Checkpoint`进行保存的情况下，也将保存所有被`Dense`层创建的变量。

当变量被分配到多个工作区时，每个工作区只写自己的checkpoint的部分，然后将这些部分合并/重新索引为一个checkpoint（实际上并没有合并，这是对外行为是一个整体），当要求所有工作区能够“看到”一个共同的文件系统时，这有效地避免了为一个工作区复制所有变量。

`tf.keras.Model.save_weights`和`tf.train.Checkpoint.save`以相同的格式保存，请注意，结果checkpoint的根是save方法所属的对象。这意味着使用`save_weights`保存`tf.keras.Model`的变量和通过`tf.train.Checkpoint`保存`Model`的变量讲不一样（反之，载入的model也不一样）。

+ 参数
   + **kwargs：关键字参数，将被设置为这个对象的属性，并保存在checkpoint中，值必须是可跟踪的对象。

+ 异常
   + ValueError：如果传入的对象不是可跟踪对象，将会引发的错误。

+ 属性
   + save_counter：当`save()`被调用时，如果可跟踪对象在训练的过程中增量式优化，则用于计数checkpoints。

### 方法
#### read

```python
read(
    save_path, options=None
)
```

+ 读取一个用`write`写入的已训练的checkpoint
+ 读取对应的`Checkpoint`已经其所有依赖的对象
+ 这个方法有点像`restore()`，但是并不指望checkpoint中的save_counter变量，它只恢复checkpoint已有所依赖的对象
+ 该方法主要用于通过使用更高级别的checkpoint管理工具，来计数和跟踪checkpoint，注意了，相对应的要使用`write()` 而不是`save()`。

示例：

```python
# 使用write()创建一个checkpoint
ckpt = tf.train.Checkpoint(v=tf.Variable(1.))
path = ckpt.write('/tmp/my_checkpoint')

# 接着使用read()加载checkpoint，这里如果使用assert_consumed()将会报错
checkpoint.read(path).assert_consumed()

# 你同样可以传递可选项给restore()
# 例如，在localhost上运行IO
options = tf.CheckpointOptions(experimental_io_device="/job:localhost")
checkpoint.read(path, options=options)
```

+ 参数
   + save_path：`write`返回的checkpoint路径
   + options：（可选）`tf.train.CheckpointOptions`对象
+ 返回：负载状态对象，其可用于做出关于checkpoint恢复的状态的断言

#### restore

```python
restore(
    save_path, options=None
)
```
+ 恢复训练的checkpoint
+ 恢复Checkpoint依赖的所有对象
+ 该方法主要用来加载通过`save()`创建的checkpoint
+ `restore()`如果要恢复已经被创建的变量，要么立即赋值，或推迟恢复直到创建的变量。如果它们具有在checkpoint的相应对象，添加依赖将会在这个调用后进行匹配（恢复请求将添加到，所有正在等候添加预期依赖性的可跟踪对象队列中）。
+ 使用`restore()`返回的状态对象的`assert_consumed()`方法，确保加载完成后没有分配的工作区未完成

```python
checkpoint = tf.train.Checkpoint( ... )
checkpoint.restore(path).assert_consumed()

# 你可以选择性的传递options给restore():
options = tf.CheckpointOptions(experimental_io_device="/job:localhost")
checkpoint.restore(path, options=options).assert_consumed()
```

如果在依赖关系图中，有任何Python对象在checkpoint中未发现，或者如果有checkpoint的值没有匹配的Python对象的，将引发异常。

+ 参数
    + save_path：checkpoint路径
    + options：（可选）`tf.train.CheckpointOptions`对象
+ 返回
   + `assert_consumed()`：如果存在变量不匹配，将会引发异常。
   + `assert_existing_objects_matched()`：如果在依赖图中，存在Python对象不匹配，引发异常。
   + `assert_nontrivial_match()`：断言匹配根对象以外的对象。
   + `expect_partial()`：有关checkpoint还原不完整的警告


#### save

```python
save(
    file_prefix, options=None
)
```
+ 保存训练的checkpoint，并提供基本的checkpoint管理。
+ 保存的checkpoint包括由该对象创建的变量和所有可追踪对象。
+ `save`是围绕`write`的基本封装，使用`save_counter`进行checkpoint编号，使用`tf.train.latest_checkpoint`更新元数据。

```python
step = tf.Variable(0, name="step")
checkpoint = tf.Checkpoint(step=step)
checkpoint.save("/tmp/ckpt")

# Later, read the checkpoint with restore()
checkpoint.restore("/tmp/ckpt").assert_consumed()

# You can also pass options to save() and restore(). For example this
# runs the IO ops on the localhost:
options = tf.CheckpointOptions(experimental_io_device="/job:localhost")
checkpoint.save("/tmp/ckpt", options=options)

# Later, read the checkpoint with restore()
checkpoint.restore("/tmp/ckpt", options=options).assert_consumed()
```

+ 参数
   + file_prefix：checkpoint文件名的前缀，即checkpoint文件名是基于它和`Checkpoint.save_counter`组合而成的
   + options：（可选）`tf.train.CheckpointOptions`对象
+ 返回：checkpoint的完整路径


#### write

```python
write(
    file_prefix, options=None
)
```

+ 写入训练checkpoint
+ 保存的checkpoint包括由该对象创建的变量和所有可追踪对象。
+ `write`不进行checkpoint编号，增量`save_counter`或使用`tf.train.latest_checkpoint`更新元数据。它主要使用更高级的checkpoint管理工具，`save`只是提供了这些特性非常基础的实现。
+ 使用`write`生成的checkpoint，必须使用`read`读取

```python
step = tf.Variable(0, name="step")
checkpoint = tf.Checkpoint(step=step)
checkpoint.write("/tmp/ckpt")

# Later, read the checkpoint with read()
checkpoint.read("/tmp/ckpt").assert_consumed()

# You can also pass options to write() and read(). For example this
# runs the IO ops on the localhost:
options = tf.CheckpointOptions(experimental_io_device="/job:localhost")
checkpoint.write("/tmp/ckpt", options=options)

# Later, read the checkpoint with read()
checkpoint.read("/tmp/ckpt", options=options).assert_consumed()
```

+ 参数
   + file_prefix：checkpoint文件名的前缀
   + options：（可选）`tf.train.CheckpointOptions`对象
+ 返回：checkpoint的完整路径