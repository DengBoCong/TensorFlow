## tf.math.argmax | tf.argmax

返回张量轴上具有最大值的索引

```python
tf.math.argmax(
    input, axis=None, output_type=tf.dtypes.int64, name=None
)
```
```python
如果相同则返回最小索引

A = tf.constant([2, 20, 30, 3, 6])
tf.math.argmax(A)
# <tf.Tensor: shape=(), dtype=int64, numpy=2>

B = tf.constant([[2, 20, 30, 3, 6], [3, 11, 16, 1, 8],
          [14, 45, 23, 5, 27]])
tf.math.argmax(B, 0)
# <tf.Tensor: shape=(5,), dtype=int64, numpy=array([2, 2, 0, 2, 2])>

C = tf.constant([0, 0, 0, 0])
tf.argmax(C)
# <tf.Tensor: shape=(), dtype=int64, numpy=0>
```


+ 参数
   + input：一个张量
   + axis：一个整数，沿该轴进行计算，默认为0
   + output_type：（可选）输出类型， (tf.int32或tf.int64)，默认tf.int64
   + name：（可选）操作名称
+ 返回值：一个张量