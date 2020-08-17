## 引言
在聊GradientTape之前，我们不得不提一下自动微分技术，要知道在自动微分技术之前，机器学习社区中很少发挥这个利器，一般都是用Backpropagation(反向传播算法)进行梯度求解，然后使用SGD等进行优化更新。梯度下降法（Gradient Descendent）是机器学习的核心算法之一，自动微分则是梯度下降法的核心，梯度下降是通过计算参数与损失函数的梯度并在梯度的方向不断迭代求得极值；

## GradientTape
GradientTape在模型中使用是，一般配合Optimizer使用，优化器有一个方法`apply_gradients`()用于优化梯度

tensorflow 提供`tf.GradientTape` api来实现自动求导功能。只要在`tf.GradientTape()`上下文中执行的操作，都会被记录与“tape”中，然后tensorflow使用反向自动微分来计算相关操作的梯度。可训练变量（由`tf.Variable`或创建`tf.compat.v1.get_variable`，`trainable=True`在两种情况下均为默认值）将被自动监视。通过watch在此上下文管理器上调用方法，可以手动监视张量。

例如，考虑函数`y = x * x`，`x = 3.0`时的梯度可以计算为：

```python
import tensorflow as tf

x = tf.constant(3.0)
with tf.GradientTape() as tape:
	tape.watch(x)
	y = x * x
dy_dx = tape.gradient(y, x) # 求微分 tf.Tensor(6.0, shape=(), dtype=float32)
```

可以嵌套GradientTapes来计算高阶导数，如下

```python
x = tf.constant(2.0)
with tf.GradientTape() as tape:
  tape.watch(x)
  with tf.GradientTape() as tt:
    tt.watch(x)
    y = x * x
  dy_dx = tt.gradient(y, x) # tf.Tensor(4.0, shape=(), dtype=float32)
dy2_dx2 = tape.gradient(dy_dx, x) # tf.Tensor(2.0, shape=(), dtype=float32)
```

默认情况下，只要调用`GradientTape.gradient()`方法，就会释放GradientTape拥有的资源, 要在同一计算上计算多个梯度，请创建一个持久梯度带。 当tape对象被垃圾回收释放资源时，这允许多次调用`gradient()`方法。 例如：

```python
import tensorflow as tf

x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as tape:
  tape.watch(x)
  y = x * x
  z = y * y
dz_dx = tape.gradient(z, x) # tf.Tensor(108.0, shape=(), dtype=float32)
dy_dx = tape.gradient(y, x) # tf.Tensor(6.0, shape=(), dtype=float32)

del tape # 删除tape引用
```

默认情况下，GradientTape将自动监视在上下文中访问的所有可训练变量， 如果要对监视哪些变量进行精细控制，可以通过将`watch_accessed_variables = False`传递给tape构造函数来禁用自动跟踪：

```python
with tf.GradientTape(watch_accessed_variables=False) as tape:
  tape.watch(variable_a)
  y = variable_a ** 2  # 梯度作用于`variable_a`.
  z = variable_b ** 3  # 由于`variable_b` 没有被watch，所有不会计算梯度
```

请注意，在使用模型时，应确保在使用`watch_accessed_variables = False`时变量存在，否则，这将导致你的迭代中没有使用梯度：

```python
a = tf.keras.layers.Dense(32)
b = tf.keras.layers.Dense(32)

with tf.GradientTape(watch_accessed_variables=False) as tape:
  tape.watch(a.variables)  # 由于此时尚未调用`a.build`，因此`a.variables`将返
  							# 回一个空列表，并且tape将不会监视任何内容。
  result = b(a(inputs))
  tape.gradient(result, a.variables)  # 该计算的结果将是“None”的列表，因为不会监视a的变量
```

请注意，只有具有实型或复杂dtype的张量才是可微的

+ 参数
   + persistent：Boolean，用于控制是否创建持久梯度带，默认情况下为False，这意味着最多可以在此对象上对gradient()方法进行一次调用
   + watch_accessed_variables：Boolean，控制tape在处于活动状态时是否将自动监视所有（可训练的）变量，默认值为True，表示可以从tape中通过计算可训练变量得出的结果来计算梯度。 如果False用户必须显式地`watch`他们想要从中计算梯度的变量

## 方法
#### batch_jacobian

```python
batch_jacobian(
    target, source, unconnected_gradients=tf.UnconnectedGradients.NONE,
    parallel_iterations=None, experimental_use_pfor=True
)
```
Jacobian（雅克比定律），有关雅可比定律的定义自行网上查看。 此函数实质如下：

```python
tf.stack([self.jacobian(y[i], x[i]) for i in range(x.shape[0])])
```

下面是使用示例：

```python
with tf.GradientTape() as g:
  x = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
  g.watch(x)
  y = x * x
batch_jacobian = g.batch_jacobian(y, x)
# batch_jacobian is [[[2,  0], [0,  4]], [[6,  0], [0,  8]]]
```
+ 参数
   + target：2D张量或更高且形状为`[b，y1，...，y_n]`的张量。 `target[i,...]`只依赖于`source[i,...]`
   + source：2D张量或更高且形状为`[b，x1，...，x_m]`的张量。
   + unconnected_gradients：一个值，可以保留“none”或“零”，并更改目标和源未连接时将返回的值。
   + parallel_iterations：一个旋钮，用于控制并行调度的迭代次数，该旋钮可用于控制总内存使用量。
   + experimental_use_pfor：如果为true，则使用pfor计算雅可比行列式。其他使用`tf.while_loop`进行计算。
+ 返回值：`t`形状为`[b，y_1，...，y_n，x1，...，x_m]` 的张量。

#### gradient

```python
gradient(
    target, sources, output_gradients=None,
    unconnected_gradients=tf.UnconnectedGradients.NONE
)
```
使用在此tape上下文中记录的操作计算梯度。
+ 参数
   + target：张量或者变量的列表或者嵌套结构
   + sources：张量或者变量的列表或者嵌套结构。
   + output_gradients：梯度的列表，一一对应target中的元素，默认为None。
   + unconnected_gradients：一个值，可以保留“none”或“zero”，并更改目标和源未连接时将返回的值。
+ 返回值：张量的列表或嵌套结构（或IndexedSlices或None），一一对应sources中的元素，返回的结构与的sources相同。

#### jacobian

```python
jacobian(
    target, sources, unconnected_gradients=tf.UnconnectedGradients.NONE,
    parallel_iterations=None, experimental_use_pfor=True
)
```
使用在此tape上下文中记录的操作来计算jacobian。

```python
with tf.GradientTape() as g:
  x  = tf.constant([1.0, 2.0])
  g.watch(x)
  y = x * x
jacobian = g.jacobian(y, x)
# jacobian value is [[2., 0.], [0., 4.]]
```

+ 参数
   + target：张量。
   + sources：张量或变量的列表或嵌套结构。
   + unconnected_gradients：一个值，可以保留“none”或“zero”，并更改目标和源未连接时将返回的值。
   + parallel_iterations：一个旋钮，用于控制并行调度的迭代次数。该旋钮可用于控制总内存使用量。
   + experimental_use_pfor：如果为true，则使用pfor计算雅可比行列式。其他使用`tf.while_loop`进行计算。
+ 返回值：张量的列表或嵌套结构（或无），一一对应sources中的元素。
+ 
#### reset

```python
reset()
```
清除此tape中存储的所有信息。等效于退出并重新进入tape上下文管理器。例如，以下两个代码块是等效的：

```python
with tf.GradientTape() as t:
  loss = loss_fn()
with tf.GradientTape() as t:
  loss += other_loss_fn()
t.gradient(loss, ...)  # Only differentiates other_loss_fn, not loss_fn


# The following is equivalent to the above
with tf.GradientTape() as t:
  loss = loss_fn()
  t.reset()
  loss += other_loss_fn()
t.gradient(loss, ...)  # Only differentiates other_loss_fn, not loss_fn
```
如果你不想退出tape的上下文管理器，或者因为所需的重置点在控制流构造内部而不能退出，这将非常有用：

```python
with tf.GradientTape() as t:
  loss = ...
  if loss > k:
    t.reset()
```
#### stop_recording

```python
@tf_contextlib.contextmanager
stop_recording()
```
暂时停止在该tape上进行记录操作。此上下文管理器处于活动状态时执行的操作不会记录在tape上。这对于减少通过跟踪所有计算而使用的内存很有用。

```python
with tf.GradientTape(persistent=True) as t:
    loss = compute_loss(model)
    with t.stop_recording():
      # The gradient computation below is not traced, saving memory.
      grads = t.gradient(loss, model.variables)
```
#### watch

```python
watch(
    tensor
)
```
确认被tape跟踪的张量
+ 参数：
   + tensor：张量或张量列表。

#### watched_variables
按构造顺序返回此tape监视的变量。
```python
watched_variables()
```
#### __enter__
输入一个上下文，在该上下文中将操作记录在此tape上。

```python
__enter__()
```
#### __exit__
退出记录上下文，不再跟踪其他操作。

```python
__exit__(
    typ, value, traceback
)
```
