## tf.convert_to_tensor

将给定值转换为张量

```python
tf.convert_to_tensor(
    value, dtype=None, dtype_hint=None, name=None
)
```

该函数将各种类型的Python对象转换为Tensor对象，它接受Tensor对象，numpy数组，Python列表和Python标量，例如

```python
def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg

value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
print(value_1)
# tf.Tensor(
#   [[1. 2.]
#   [3. 4.]], shape=(2, 2), dtype=float32)

value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
print(value_2)
# tf.Tensor(
#   [[1. 2.]
#   [3. 4.]], shape=(2, 2), dtype=float32)

value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
print(value_3)
# tf.Tensor(
#   [[1. 2.]
#   [3. 4.]], shape=(2, 2), dtype=float32)
```

在Python中编写新操作时（例如上例中的my_func），此函数很有用。所有标准的Python op构造函数都将此函数应用于其每个Tensor值输入，这使这些op除了Tensor对象还接受numpy数组，Python列表和标量。

+ 参数
   + value：类型具有已注册的Tensor转换方法的对象
   + dtype：返回的张量的可选元素类型。如果缺少，则从值的类型推断出类型
   + dtype_hint：返回的张量的可选元素类型，当dtype为None时使用。在某些情况下，调用者在转换为张量时可能没有dtype的想法，因此dtype_hint可用作软优先级。如果无法转换为dtype_hint，则此参数无效
   + name：（可选）当新的Tensor被创建时使用
+ 返回值：基于value的张量