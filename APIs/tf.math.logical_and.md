## tf.math.logical_and | tf.logical_and

逻辑与方法

```python
tf.math.logical_and(
    x, y, name=None
)
```
该操作适用于以下输入类型：
+ 布尔型的两个单个元素
+ 一个bool类型的tf.Tensor和一个bool，将会把单个bool与Tensor中的每个元素进行逻辑与操作
+ 两个bool类型的tf.Tensor，在这种情况下，结果将是两个输入张量的按元素逻辑与
```python
a = tf.constant([True])
b = tf.constant([False])
tf.math.logical_and(a, b)
<tf.Tensor: shape=(1,), dtype=bool, numpy=array([False])>
```

```python
c = tf.constant([True])
x = tf.constant([False, True, True, False])
tf.math.logical_and(c, x)
<tf.Tensor: shape=(4,), dtype=bool, numpy=array([False, True, True, False])>
```


```python
y = tf.constant([False, False, True, True])
z = tf.constant([False, True, False, True])
tf.math.logical_and(y, z)
<tf.Tensor: shape=(4,), dtype=bool, numpy=array([False, False, False, True])>
```

+ 参数
   + x：bool张量
   + y：bool张量
   + name：（可选）操作名称
+ 返回值：张量