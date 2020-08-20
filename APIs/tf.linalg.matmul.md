## tf.linalg.matmul


将矩阵`a`乘以矩阵`b`，得到`a` * `b`。

```python
tf.linalg.matmul(
    a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False,
    a_is_sparse=False, b_is_sparse=False, name=None
)
```
在进行任何换位，输入必须是 秩>= 2的张量，其中内部2个维指定有效的矩阵乘法维，而任何其他外部维指定匹配的批量大小。

两种矩阵必须属于同一类型。支持的类型为：`float16`，`float32`，`float64`，`int32`，`complex64`，`complex128`

通过将相应标志之一设置为True，可以即时对矩阵进行转置或连接（共轭和转置），这些默认为False

如果一个或两个矩阵都包含大量零，则可以通过将相应的`a_is_sparse`或`b_is_sparse`标志设置为True来使用更有效的乘法算法。这些默认为False。此优化仅适用于数据类型为`bfloat16`或`float32`的普通矩阵（秩2张量）。

一个简单的二维张量矩阵乘法：

```python
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
a
# <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
# array([[1, 2, 3],
#    [4, 5, 6]], dtype=int32)>

b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
b
# <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
# array([[ 7,  8],
#    [ 9, 10],
#    [11, 12]], dtype=int32)>

c = tf.matmul(a, b)
c
# <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
# array([[ 58,  64],
#    [139, 154]], dtype=int32)>
```
一个批量矩阵乘法，批量形状为[2]：

```python
a = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])
a
# <tf.Tensor: shape=(2, 2, 3), dtype=int32, numpy=
# array([[[ 1,  2,  3],
#    [ 4,  5,  6]],
#    [[ 7,  8,  9],
#    [10, 11, 12]]], dtype=int32)>
b = tf.constant(np.arange(13, 25, dtype=np.int32), shape=[2, 3, 2])
b
# <tf.Tensor: shape=(2, 3, 2), dtype=int32, numpy=
# array([[[13, 14],
#    [15, 16],
#    [17, 18]],
#    [[19, 20],
#    [21, 22],
#    [23, 24]]], dtype=int32)>

c = tf.matmul(a, b)
c
# <tf.Tensor: shape=(2, 2, 2), dtype=int32, numpy=
# array([[[ 94, 100],
#      [229, 244]],
#      [[508, 532],
#      [697, 730]]], dtype=int32)>
```

由于python> = 3.5，因此支持@运算符。在TensorFlow中，它仅调用tf.matmul()函数，因此以下几行是等效的
```python
d = a @ b @ [[10], [11]]
d = tf.matmul(tf.matmul(a, b), [[10], [11]])
```

+ 参数
   + a：秩大于1的张量，支持类型有float16, float32, float64, int32, complex64, complex128
   + b：和a有着相同类型和秩的张量
   + transpose_a：如果为True，则在相乘之前先转置a
   + transpose_b：如果为True，则在相乘之前先转置b
   + adjoint_a：如果为True，则在乘法之前对a进行共轭和转置
   + adjoint_b：如果为True，则在乘法之前对b进行共轭和转置
   + a_is_sparse：如果为True，则将a视为稀疏矩阵。注意，这不支持`tf.sparse.SparseTensor`，它只是进行优化，假设a中的大多数值为零。有关`tf.sparse.SparseTensor`乘法的某些支持，请参见`tf.sparse.sparse_dense_matmul`
   + b_is_sparse：如果为True，则将b视为稀疏矩阵。注意，这不支持`tf.sparse.SparseTensor`，它只是进行优化，假设b中的大多数值为零。有关`tf.sparse.SparseTensor`乘法的某些支持，请参见`tf.sparse.sparse_dense_matmul`
   + name：（可选）操作名称
+ 返回值：与a和b类型相同的张量，其中每个最里面的矩阵是a和b中对应矩阵的乘积，例如如果所有转置或伴随属性均为False：
`output[..., i, j] = sum_k(a[..., i, k] * b[..., k, j])`