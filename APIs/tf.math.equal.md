## tf.math.equal

返回（x == y）的真值

```python
tf.math.equal(
    x, y, name=None
)
```

使用参数执行broadcast机制，然后进行逐元素的相等比较，返回布尔值的张量
```python
x = tf.constant([2, 4])
y = tf.constant(2)
tf.math.equal(x, y)
# <tf.Tensor: shape=(2,), dtype=bool, numpy=array([ True, False])>
```

```python
x = tf.constant([2, 4])
y = tf.constant([2, 4])
tf.math.equal(x, y)
# <tf.Tensor: shape=(2,), dtype=bool, numpy=array([ True,  True])>
```

+ 参数
   + x：tf.Tensor、tf.sparse.SparseTensor或者tf.IndexedSlices
   + y：tf.Tensor、tf.sparse.SparseTensor或者tf.IndexedSlices
   + name：（可选）操作名称
+ 返回值：类型为bool的tf.tensor，与x或y具有相同的大小