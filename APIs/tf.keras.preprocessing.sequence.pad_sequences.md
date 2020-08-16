## pad_sequences（填充序列）

```python
keras.preprocessing.sequence.pad_sequences(
  sequences, 
  maxlen=None, 
  dtype='int32', 
  padding='pre', 
  truncating='pre', 
  value=0.
)

```

将长为nb_samples的序列（标量序列）转化为形如`(nb_samples,nb_timesteps)`的2D numpy array。除非提供了参数`maxlen，nb_timesteps=maxlen`，否则其值为最长序列的长度，其他短于该长度的序列都会在后部填充0以达到该长度。而长于nb_timesteps的序列将会被截断，以使其匹配目标长度。`padding`和截断发生的位置分别取决于`padding`和`truncating`

+ 参数
  + sequences：浮点数或整数构成的两层嵌套列表
  + maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0.
  + dtype：返回的numpy array的数据类型
  + padding：‘pre’或‘post’，确定当需要补0时，在序列的起始还是结尾补
  + truncating：‘pre’或‘post’，确定当需要截断序列时，从起始还是结尾截断
  + value：浮点数，此值将在填充时代替默认的填充值0
+ 返回值
  + 返回形如(nb_samples,nb_timesteps)的2D张量


## 示例

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

sequence = [[1], [2, 3], [4, 5, 6]]
print(pad_sequences(sequence))
# [[0 0 1]
# [0 2 3]
# [4 5 6]]

print(pad_sequences(sequence, value=-1))
# [[-1 -1  1]
# [-1  2  3]
# [ 4  5  6]]

print(pad_sequences(sequence, padding='post'))
# [[1 0 0]
# [2 3 0]
# [4 5 6]]

print(pad_sequences(sequence, maxlen=2))
# [[0 1]
# [2 3]
# [5 6]]



```