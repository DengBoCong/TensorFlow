## tf.size

```python
tf.size(
    input, out_type=tf.dtypes.int32, name=None
)
```

返回一个0D张量，表示类型为out_type的输入中的元素数。默认为`tf.int32`

```python
t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
tf.size(t)
# <tf.Tensor: shape=(), dtype=int32, numpy=12>
```

+ 参数
   + input：Tensor或SparseTensor
   + name：（可选）操作名称
   + out_type：（可选）操作的指定非量化数字输出类型，默认为tf.int32
+ 返回值：张量，默认tf.int32