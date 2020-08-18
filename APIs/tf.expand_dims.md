## tf.expand_dims

返回在索引轴上插入长度为1的轴的张量

```python
tf.expand_dims(
    input, axis, name=None
)
```

给定张量输入，此操作将在输入形状的维度索引轴上插入长度为1的维度。维度索引遵循Python索引规则：从零开始，如果它是一个负索引，从末尾算起。这个操作在如下情况中非常有用：

+ 添加“batch”轴
+ 对齐轴进行广播。
+ 将内部向量长度轴添加到标量张量


## 示例
如果有一张形状为`[height, width, channels]`的图像

```python
image = tf.zeros([10,10,3])
```
你可以通过传递`axis= 0`来添加外部批处理轴：
```python
tf.expand_dims(image, axis=0).shape.as_list()
# [1, 10, 10, 3]
```
新的轴位置与Python `list.insert(axis，1)`一致
```python
tf.expand_dims(image, axis=1).shape.as_list()

# [10, 1, 10, 3]
```
此操作要求轴是遵循Python索引规则,即是`input.shape`的有效索引
```python
-1-tf.rank(input) <= axis <= tf.rank(input)
```
该方法相关操作：
+ `tf.squeeze`，用来删除维度为`1`的维度
+ `tf.reshape`，提供复杂的调整形状的能力
+ `tf.sparse.expand_dims`，专门为`tf.SparseTensor`提供的


+ 参数
   + input：一个张量
   + axis：整数，用于指定要扩展输入形状的维度索引。在输入为D维度的情况下，轴必须在[-（D + 1），D]（含）范围内
   + name：（可选）输出张量的名称
+ 返回值：具有与输入相同数据的张量，并在由轴指定的索引处插入附加维度