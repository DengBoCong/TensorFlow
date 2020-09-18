## tf.keras.layers.Lambda

将任意表达式包装为`Layer`对象，集成子`Layer`

```python
tf.keras.layers.Lambda(
    function, output_shape=None, mask=None, arguments=None, **kwargs
)
```

存在`Lambda`层，因此在构建`Sequential`和函数式API模型时可以使用任意TensorFlow方法。`Lambda`层最适合简单操作或快速实验。有关更高级的用例，请遵循此指南对`tf.keras.layers.Layer`进行子类化。

继承`tf.keras.layers.Layer`而不是使用`Lambda`层的主要原因是保存和检查模型。`Lambda`层是通过序列化Python字节码来保存的，而子类化的层可以通过覆盖其`get_config`方法来保存，覆盖`get_config`可改善模型的可移植性，依赖于子类化图层的模型通常也更易于可视化和推理。


```python
# add a x -> x^2 layer
model.add(Lambda(lambda x: x ** 2))
```

```python
# add a layer that returns the concatenation
# of the positive part of the input and
# the opposite of the negative part

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

model.add(Lambda(antirectifier))
```
尽管可以在Lambda层上使用变量，但不建议使用此方法，因为它很容易导致错误。例如：
```python
scale = tf.Variable(1.)
scale_layer = tf.keras.layers.Lambda(lambda x: x * scale)
```
由于`scale_layer`不会直接跟踪scale变量，因此它不会出现在`scale_layer.trainable_weights`中，因此，如果在模型中使用`scale_layer`，则不会对其进行训练。更好的模式是编写一个子类化的Layer
```python
 class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self):
      super(ScaleLayer, self).__init__()
      self.scale = tf.Variable(1.)

    def call(self, inputs):
      return inputs * self.scale
```
通常，Lambda层可以方便地进行简单的无状态计算，但是任何更复杂的操作都应使用子类Layer来代替。

+ 参数
   + function：要评估的方法，将输入张量作为第一个参数。
   + output_shape：方法的预期输出形状。如果未明确提供，则可以推断出该参数。可以是元组或方法。如果是元组，则仅指定向前的第一维；假定样本维数与输入的维数相同：`output_shape = (input_shape[0], ) + output_shape`输入为None，样本维度也为`None: output_shape = (None, ) + output_shape`，如果是函数，则将整个形状指定为输入形状的函数：`output_shape = f(input_shape)`
   + mask：无（表示无遮罩）或具有与layer方法相同的签名的可调用项，或者张量将作为输出掩码返回，而不管输入是什么。 compute_mask
   + arguments：传递给函数的关键字参数的可选字典。

**输入形状：**
任意。当将此层用作模型的第一层时，请使用关键字参数input_shape（整数元组，不包括示例轴）。

**输出形状：**
由output_shape参数指定