## tf.math.reduce_mean | tf.reduce_mean

`tf.math.reduce_mean`和`tf.reduce_mean`都是指向同一个方法，用于计算张量维度上元素的均值。

```python
tf.math.reduce_mean(
    input_tensor, axis=None, keepdims=False, name=None
)
```

通过计算轴上各个维度的元素平均值，沿轴上给定的维度减少input_tensor。 除非keepdims为true，否则对于轴上的每个条目，张量的秩都会减小1。 如果keepdims为true，则减小的尺寸保留为长度1。如果axis为None，则减小所有尺寸，并返回具有单个元素的张量。

```python
x = tf.constant([[1., 1.], [2., 2.]])
tf.reduce_mean(x) 
# <tf.Tensor: shape=(), dtype=float32, numpy=1.5>

tf.reduce_mean(x, 0)
# <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1.5, 1.5], dtype=float32)>

tf.reduce_mean(x, 1)
# <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 2.], dtype=float32)>

```

+ 参数
   + input_tensor：用于计算的张量，应该为数字类型
   + axis：计算的维度，如果为None（默认），则计算所有元素，必须在[-rank（(input_tensor)，rank(input_tensor)范围内。这里要理解reduce的含义，因为计算了某个轴上的元素之后，相当于直接减少了一个轴。
   + keepdims：如果为True，保持计算的轴的长度为1
   + name（可选）操作名称
+ 返回值：减少之后的张量

## 和numpy的兼容性

与np.mean等效，请注意，np.mean具有dtype参数，可用于指定输出类型，默认情况下，这是dtype = float64。 另一方面，tf.reduce_mean具有从input_tensor中主动类型推断，例如：

```python
x = tf.constant([1, 0, 1, 0])
tf.reduce_mean(x)
# <tf.Tensor: shape=(), dtype=int32, numpy=0>

y = tf.constant([1., 0., 1., 0.])
tf.reduce_mean(y)
# <tf.Tensor: shape=(), dtype=float32, numpy=0.5>
```