## tf.py_function

将python方法包装到TensorFlow op中，该方法会eager模式执行执行它。

```python
tf.py_function(
    func, inp, Tout, name=None
)
```
此方法允许将TensorFlow图中的计算表示为Python函数。特别是，它将Python函数`func`包装在可区分的TensorFlow操作中，该操作在启用了eager模式执行的情况下执行该函数。因此，`tf.py_function`使得可以使用Python构造（`if`， `while`， `for`等）代替TensorFlow控制流构造（ `tf.cond`，`tf.while_loop`）来表达控制流。例如，你可以使用`tf.py_function`来实现日志功能：

```python
def log_huber(x, m):
  if tf.abs(x) <= m:
    return x**2
  else:
    return m**2 * (1 - 2 * tf.math.log(m) + tf.math.log(x**2))

x = tf.compat.v1.placeholder(tf.float32)
m = tf.compat.v1.placeholder(tf.float32)

y = tf.py_function(func=log_huber, inp=[x, m], Tout=tf.float32)
dy_dx = tf.gradients(y, x)[0]

with tf.compat.v1.Session() as sess:
  # 会话eager模式执行`log_huber`，给定以下值，它将进入第一个分支，因此`y`的值为1.0，而dy_dx`的值为2.0
  # y, dy_dx = sess.run([y, dy_dx], feed_dict={x: 1.0, m: 2.0})
```
你还可以使用`tf.py_function`在运行时使用Python工具调试模型，即可以隔离要调试的代码部分，将其包装在Python函数中，并根据需要插入pdb跟踪点或打印语句，然后包装这些`tf.py_function`函数。

有关eager模式执行的更多信息，请参见eager模式指南。

`tf.py_function`在本质上与`tf.compat.v1.py_func`类似，但是与后者不同，前者允许你在包装的Python函数中使用TensorFlow操作。特别是，虽然`tf.compat.v1.py_func`仅在CPU上运行并包装将NumPy数组作为输入并返回NumPy数组作为输出的函数，`tf.py_function`可以放在GPU上并包装将function作为输入的函数，执行TensorFlow在其体内进行操作，并返回张量作为输出。

与`tf.compat.v1.py_func`一样，`tf.py_function`在序列化和分发方面具有以下限制：
+ 函数的主体（即`func`）将不会在`GraphDef`序列化。因此，如果需要序列化模型并在其他环境中还原它，则不应使用此功能。
+ 该操作必须在与调用`tf.py_function()`的Python程序相同的地址空间中运行。如果你使用的是分布式TensorFlow，则必须在与调用`tf.py_function()`的程序相同的过程中运行`tf.distribute.Server`，并且必须将创建的操作固定到该服务器上的设备（例如，`with tf.device()`: 。


+ 参数
   + func：一个Python函数，该函数接受具有与int中对应的tf.Tensor对象匹配的元素类型的Tensor对象的列表，并返回具有与Tout中的对应值匹配的元素类型的Tensor对象的列表（或单个Tensor或None）。
   + inp：Tensor对象列表。
   + Tout：张量流数据类型的列表或元组，或者只有一个张量流数据类型（如果只有一个，表示函数返回）； 如果没有返回值（即，如果返回值为None），则为空列表
   + name：操作的名称（可选）
+ 返回值：func计算的Tensor或单个Tensor列表；如果func返回None，则返回一个空列表。