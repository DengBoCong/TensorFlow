## tf.cast

将张量转换为新类型

```python
tf.cast(
    x, dtype, name=None
)
```

该操作将`x`（对于`Tensor`）或`x.values`（对于`SparseTensor`或`IndexedSlices`）强制转换为`dtype`

```python
x = tf.constant([1.8, 2.2], dtype=tf.float32)
tf.cast(x, tf.int32)
# <tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 2], dtype=int32)>
```

该操作支持`uint8`，`uint16`，`uint32`，`uint64`，`int8`，`int16`，`int32`，`int64`，`float16`，`float32`，`float64`，`complex64`，`complex128`，`bfloat16`的数据类型（用于x和dtype）。 如果将复杂类型（`complex64`，`complex128`）转换为实类型，则仅返回x的实部。如果将实类型转换为复杂类型（`complex64`，`complex128`），则返回值的虚部设置为`0`，此处对复杂类型的处理与numpy的行为一致。

+ 参数
   + x：tf.Tensor、tf.sparse.SparseTensor或者tf.IndexedSlices，支持类型有`uint8`，`uint16`，`uint32`，`uint64`，`int8`，`int16`，`int32`，`int64`，`float16`，`float32`，`float64`，`complex64`，`complex128`，`bfloat16`的数据类型
   + dtype：目标类型，所支持的类型和上面 一致
   + name：（可选）操作名称
+ 返回值：返回tf.Tensor、tf.sparse.SparseTensor或者tf.IndexedSlices，其与x形状相同，与dtype类型相同