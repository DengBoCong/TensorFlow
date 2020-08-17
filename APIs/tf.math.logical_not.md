## tf.math.logical_not

返回`NOT x`的真值

```python
tf.math.logical_not(
    x, name=None
)
```

```python
tf.math.logical_not(tf.constant([True, False]))
# <tf.Tensor: shape=(2,), dtype=bool, numpy=array([False,  True])>
```

+ 参数
   + x：布尔值的张量
   + name：（可选）操作名称
+ 返回值：布尔值的张量