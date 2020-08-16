## text_to_word_sequence（句子分割）

```python
keras.preprocessing.text.text_to_word_sequence(
  text, 
  filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', 
  lower=True, 
  split=" ")
```

本函数将一个句子拆分成单词构成的列表，使用filters参数中定义的标点符号和split参数中定义的分隔符作为分割句子的标准。text_to_word_sequence，将文本转换为一个字符序列，即将文本转换为序列（即单词在字典中的下标构成的列表，从1算起）。

+ 参数：
  + text：字符串，待处理的文本
  + filters：需要滤除的字符的列表或连接形成的字符串，例如标点符号。
    + 默认值为 '!"#$%&()*+,-./:;<=>?@[]^_`{|}~\t\n'，包含标点符号，制表符和换行符等。
  + lower：布尔值，是否将序列设为小写形式
  + split：字符串，单词的分隔符，如空格
	返回值：字符串列表

## 示例

```python
from tensorflow.keras.preprocessing.text import text_to_word_sequence

text="我爱你!!你爱我么??"

sequence = text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', lower=True, split=" ")
print(sequence)
# ['我爱你', '你爱我么']

text="好好学习,天天向上!!"
sequence = text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', lower=True, split=" ")
print(sequence)
# ['好好学习', '天天向上']


#使用了中文形式的标点符号，因此filters参数中也应加上中文形式的标点符号，才能正常分割句子
text="好好学习，天天向上！！"
sequence = text_to_word_sequence(text, filters='!"#$%&()*+，-./:;<=>?@[\]^_`{|}~\t\n', lower=True, split=" ")
print(sequence)
# ['好好学习', '天天向上！！']
```