## tf.shape
返回张量的形状

```python
tf.shape(
    input, out_type=tf.dtypes.int32, name=None
)
```

tf.shape返回表示输入形状的1-D整数张量

```python
t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])

tf.shape(t)

# <tf.Tensor: shape=(3,), dtype=int32, numpy=array([2, 2, 3], dtype=int32)>
```

```python
a = tf.keras.layers.Input((None, 10))
tf.shape(a)
# <tf.Tensor 'Shape:0' shape=(3,) dtype=int32>
```

在这些情况下，使用`tf.Tensor.shape`将返回更多信息
```python
a.shape
TensorShape([None, None, 10])
```

（第一个`None`表示尚不知道的批量大小）在eager模式下，`tf.shape`和`Tensor.shape`应该相同。在`tf.function`或`compat.v1`上下文中，一直到执行时间之前，并非所有尺寸都可以知道。因此，在图模式中定义自定义层和模型时，首选动态`tf.shape(x)`而不是静态`x.shape`。


+ 参数
   + input：Tensor或者SparseTensor
   + out_type：（可选）指定输出类型（int32或int64），默认为tf.int32
   + name：操作名称（可选）
+ 返回值：类型为out_type的张量