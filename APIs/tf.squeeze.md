##  tf.squeeze

移除张量的形状中，为1的维度

```python
tf.squeeze(
    input, axis=None, name=None
)
```
给定张量输入，此操作将返回相同类型的张量，并删除维度为1的所有维度。如果你不想删除所有维度为1的维度，则可以通过指定轴来删除维度为1的特定维度

```python
# 't' 是一个张量，且形状为[1, 2, 1, 3, 1, 1]
tf.shape(tf.squeeze(t))  # [2, 3]
```

或者移除指定的为1的维度
```python
# 't'是一个张量，且形状为[1, 2, 1, 3, 1, 1]
tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]
```
与旧版op 'tf.compat.v1.squeeze'不同，此op不接受过时的'squeeze_dims'参数。

+ 参数
   + input：一个张拉你个
   + axis：可选的整数列表。 默认为[]。 如果指定，则仅移除列出的维度。维度索引从0开始，移除不为1的维度是会报错的。必须在[-rank(input)，rank(input))范围内。 如果输入是RaggedTensor，则必须指定
   + name：（可选）操作名称
+ 返回值：张量，具有与输入相同的类型。包含与输入相同的数据，但删除了维度为1的一个或多个维度