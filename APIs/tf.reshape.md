## tf.reshape

tf.reshape顾名思义就是重新调整张量的形状。

```python
tf.reshape(
    tensor, shape, name=None
)
```
给定张量，此操作将返回一个新的`tf.Tensor`，其值和张量的顺序相同，但具有由shape赋予的新形状。

```python
import tensorflow as tf

t1 = [[1, 2, 3],
    [4, 5, 6]]

print(tf.shape(t1).numpy())  # [2 3]

t2 = tf.reshape(t1, [6]) # tf.Tensor([1 2 3 4 5 6], shape=(6,), dtype=int32)
print(t2)

tf.reshape(t2, [3, 2])

# <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
# array([[1, 2],
#    [3, 4],
#    [5, 6]], dtype=int32)>

```

`tf.reshape`不会更改张量中元素的顺序或总数，因此可以重用基础数据缓冲区，这使得不管操作的张量有多大，一样可以快速的运行操作。

```python
tf.reshape([1, 2, 3], [2, 2])

# Traceback (most recent call last):
# ...
# InvalidArgumentError: Input to reshape is a tensor with 3 values, but the requested shape has 4 [Op:Reshape]
```

要对数据重新排序以重新排列张量的尺寸，请参见`tf.transpose`

```python
t = [[1, 2, 3],
    [4, 5, 6]]

tf.reshape(t, [3, 2]).numpy()
# array([[1, 2],
#    [3, 4],
#    [5, 6]], dtype=int32)

tf.transpose(t, perm=[1, 0]).numpy()
# array([[1, 4],
#    [2, 5],
#    [3, 6]], dtype=int32)
```

如果形状的一个分量是特殊值-1，则将计算该尺寸的大小，以便总大小保持恒定。 特别地，[-1]的形状变平为1-D，形状的最多一个分量可以是-1

```python
t = [[1, 2, 3],
    [4, 5, 6]]

tf.reshape(t, [-1])
# <tf.Tensor: shape=(6,), dtype=int32, numpy=array([1, 2, 3, 4, 5, 6], dtype=int32)>

tf.reshape(t, [3, -1])
# <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
# array([[1, 2],
#    [3, 4],
#    [5, 6]], dtype=int32)>

tf.reshape(t, [-1, 2])
# <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
# array([[1, 2],
#    [3, 4],
#    [5, 6]], dtype=int32)>
```

`tf.reshape(t, [])`将具有一个元素的张量`t`调整为标量。

```python
tf.reshape([7], []).numpy() # 7
```
## 更多示例
```python
t = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(tf.shape(t).numpy()) # [9]
tf.reshape(t, [3, 3])
# <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
# array([[1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]], dtype=int32)>
```
```python
t = [[[1, 1], [2, 2]],
     [[3, 3], [4, 4]]]
print(tf.shape(t).numpy()) # [2 2 2]
tf.reshape(t, [2, 4])
# <tf.Tensor: shape=(2, 4), dtype=int32, numpy=
# array([[1, 1, 2, 2],
#    [3, 3, 4, 4]], dtype=int32)>
```
```python
t = [[[1, 1, 1],
      [2, 2, 2]],
     [[3, 3, 3],
      [4, 4, 4]],
     [[5, 5, 5],
      [6, 6, 6]]]

print(tf.shape(t).numpy()) # [3 2 3]
tf.reshape(t, [-1])
# <tf.Tensor: shape=(18,), dtype=int32, numpy=array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6], dtype=int32)>

tf.reshape(t, [2, -1])
# <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
# array([[1, 1, 1, 2, 2, 2, 3, 3, 3],
#    [4, 4, 4, 5, 5, 5, 6, 6, 6]], dtype=int32)>

tf.reshape(t, [-1, 9])
# <tf.Tensor: shape=(2, 9), dtype=int32, numpy=
# array([[1, 1, 1, 2, 2, 2, 3, 3, 3],
#    [4, 4, 4, 5, 5, 5, 6, 6, 6]], dtype=int32)>

tf.reshape(t, [ 2, -1, 3])
# <tf.Tensor: shape=(2, 3, 3), dtype=int32, numpy=
# array([[[1, 1, 1],
#     [2, 2, 2],
#     [3, 3, 3]],
#     [[4, 4, 4],
#     [5, 5, 5],
#     [6, 6, 6]]], dtype=int32)>
```

+ 参数
   + tensor：一个张量
   + shape：一个张量，必须是以下类型之一：int32，int64，用于定义输出张量的形状
   + name：（可选）字符串，操作名称
+ 返回值：返回一个有着同样类型的张量