## tf.transpose

用于计算`a`的转置，其中`a`是张量

```python
tf.transpose(
    a, perm=None, conjugate=False, name='transpose'
)
```

根据`perm`的值排列尺寸，返回的张量的维度`i`将对应于输入维度`perm [i]`，如果未给出`perm`，则将其设置为`(n-1 ... 0)`，其中`n`是输入张量的等级。因此，默认情况下，此操作会在2D输入张量上执行常规矩阵转置。如果`conjugate `是`True`，而`a.dtype`是`complex64`或`complex128`，则将`a`的值进行共轭和转置。

## 示例
```python
x = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.transpose(x)
# <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
# array([[1, 4],
#    [2, 5],
#    [3, 6]], dtype=int32)>
```
同样，你可以调用`tf.transpose(x，perm = [1，0])`，如果`x`是复数，则设置`conjugate = True`给出共轭转置：

```python
x = tf.constant([[1 + 1j, 2 + 2j, 3 + 3j],
          [4 + 4j, 5 + 5j, 6 + 6j]])

tf.transpose(x, conjugate=True)
# <tf.Tensor: shape=(3, 2), dtype=complex128, numpy=
# array([[1.-1.j, 4.-4.j],
#    [2.-2.j, 5.-5.j],
#    [3.-3.j, 6.-6.j]])>
```
'perm'对于n>2的n维张量更有用
```python
x = tf.constant([[[ 1,  2,  3],
          [ 4,  5,  6]],
          [[ 7,  8,  9],
          [10, 11, 12]]])
```
如上所述，简单地调用`tf.transpose`则默认`perm = [2,1,0]`。要对维度为`0`的矩阵进行转置时（例如，当对矩阵进行转置时，批次为`0`），则需要设置`perm = [0,2,1]`。

```python
x = tf.constant([[[ 1,  2,  3],
          [ 4,  5,  6]],
          [[ 7,  8,  9],
          [10, 11, 12]]])
tf.transpose(x, perm=[0, 2, 1])
# <tf.Tensor: shape=(2, 3, 2), dtype=int32, numpy=
# array([[[ 1,  4],
#    [ 2,  5],
#    [ 3,  6]],
#    [[ 7, 10],
#    [ 8, 11],
#    [ 9, 12]]], dtype=int32)>
```

+ 参数
   + a：一个张量
   + perm：a的尺寸的排列，应该是一个向量（可以理解为对初始顺序的标记，然后进行任意排序）
   + conjugate：布尔值（可选），将其设置为`True`，在数学上等效于`tf.math.conj(tf.transpose（input))`
   + name：（可选）操作名称
+ 返回值：转置后的张量

## 与numpy差异

在numpy中，转置是恒定时间的高效内存的操作，因为它们仅以调整的步幅返回相同数据的新视图，TensorFlow不支持strides，因此转置会返回一个新的张量，其中的每个项都经过排列。