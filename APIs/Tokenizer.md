## 引言
Keras的Tokenizer是一个分词器，用于文本预处理，序列化，向量化等。在我们的日常开发中，我们经常会遇到相关的概念，即token-标记、tokenize--标记化以及tokenizer--标记解析器。Tokenizer类允许通过将每个文本转换为整数序列（每个整数是字典中标记的索引）或转换成矢量（其中每个标记的系数可以是二进制的）的矢量化语料库，基于单词数 ，基于TF-IDF等等。形如如下使用创建方式：

```python
tf.keras.preprocessing.text.Tokenizer(
    num_words=None, 
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
    lower=True,
    split=' ', 
    char_level=False, 
    oov_token=None, 
    document_count=0, 
    **kwargs
)
```

## 参数

+ num_words：根据单词频率排序，保留前num_words个单词，即仅保留最常见的`num_words-1`个单词
+ filters: 一个用于过滤的正则表达式的字符串，这个过滤器作用在每个元素上，默认过滤除‘`’字符外的所有标点符号，制表符和换行符
+ lower：boolean值， 标记是否将文本转换为小写
+ split：字符串值， 分词分隔符
+ char_level：如果为True，则进行字符级别的分词
+ oov_token：如果给出的话，它将被添加到word_index中，并用于在text_to_sequence调用期间替换词汇外的单词，即用来补充原文本中没有的词。

默认情况下，将删除所有标点符号，从而将文本转换为以空格分隔的单词序列（单词可能包含'字符，如I'am）， 然后将这些序列分为标记列表，并将它们编入索引或向量化。注意力， 0是保留索引，不会分配给任何单词。


所以科学使用Tokenizer的方法是，首先用Tokenizer的 fit_on_texts 方法学习出文本的字典，然后word_index 就是对应的单词和数字的映射关系dict，通过这个dict可以将每个string的每个词转成数字，可以用texts_to_sequences，这是我们需要的，然后通过padding的方法补成同样长度，在用keras中自带的embedding层进行一个向量化，并输入到LSTM中。

## 方法

+ fit_on_texts(texts) ：
   + 参数 texts：要用以训练的文本列表。
   + 返回值：无。

+ texts_to_sequences(texts) ：
   + 参数 texts：待转为序列的文本列表。
   + 返回值：序列的列表，列表中每个序列对应于一段输入文本。

+ texts_to_sequences_generator(texts) ：
   + 本函数是texts_to_sequences的生成器函数版。
   + 参数 texts：待转为序列的文本列表。
   + 返回值：每次调用返回对应于一段输入文本的序列。

+ texts_to_matrix(texts, mode) ：
   + 参数 texts：待向量化的文本列表。
   + 参数 mode：'binary'，'count'，'tfidf'，'freq' 之一，默认为 'binary'。
   + 返回值：形如(len(texts), num_words) 的numpy array。

+ fit_on_sequences(sequences) ：
   + 参数 sequences：要用以训练的序列列表。
   + 返回值：无

+ sequences_to_matrix(sequences) ：
   + 参数 sequences：待向量化的序列列表。
   + 参数 mode：'binary'，'count'，'tfidf'，'freq' 之一，默认为 'binary'。
   + 返回值：形如(len(sequences), num_words) 的 numpy array。

+ get_config:
   + 将标记器的配置返回为Python字典，标记器使用的字数字典被序列化为纯JSON，以便其他项目可以读取配置
   + 返回值：带有tokenizer配置的Python字典

+ to_json：
   + 返回包含标记器配置的JSON字符串，要从JSON字符串加载标记器，请使用`keras.preprocessing.text.tokenizer_from_json(json_string)`。
   + 返回值：包含标记器配置的JSON字符串

## 属性
+ word_counts: 字典，将单词（字符串）映射为它们在训练期间出现的次数。仅在调用fit_on_texts之后设置。
+ word_docs: 字典，将单词（字符串）映射为它们在训练期间所出现的文档或文本的数量。仅在调用fit_on_texts之后设置。
+ word_index: 字典，将单词（字符串）映射为它们的排名或者索引。仅在调用fit_on_texts之后设置。
+ document_count: 整数。分词器被训练的文档（文本或者序列）数量。仅在调用fit_on_texts或fit_on_sequences之后设置

## 示例

```python
from tf.keras.preprocessing.text import Tokenizer
# Using TensorFlow backend.

#  创建分词器 Tokenizer 对象
tokenizer = Tokenizer()

#  text
text = ["今天 北京 下 雨 了", "我 今天 加班"]

#  fit_on_texts 方法
tokenizer.fit_on_texts(text)

#  word_counts属性
tokenizer.word_counts
# OrderedDict([('今天', 2),
#              ('北京', 1),
#              ('下', 1),
#              ('雨', 1),
#              ('了', 2),
#              ('我', 1),
#              ('加班', 1)])

#  word_docs属性
tokenizer.word_docs
# defaultdict(int, {'下': 1, '北京': 1, '今天': 2, '雨': 1, '了': 2, '我': 1, '加班': 1})

#  word_index属性
tokenizer.word_index
# {'今天': 1, '了': 2, '北京': 3, '下': 4, '雨': 5, '我': 6, '加班': 7}

#  document_count属性
tokenizer.document_count
# 2
```

需要注意的点是，由于书写习惯，英文文本的单词之间是用空格隔开的，`split=' '` 这个参数可以直接对英文文本进行空格分词。但是对中文不行，因此使用 `tokenizer.fit_on_texts(text)` 时，text如果是英文文本，可以直接 `text = ["Today is raining.", "I feel tired today."]` ，但是text是中文文本的话，需要先将中文文本分词再作为输入`text： text = ["今天 北京 下 雨 了", "我 今天 加班"]`，因此，keras的Tokenizer对于英文文档可以做`分词+嵌入` 两步，对于中文的话，其实只有`嵌入`这步。如下：

```python
from tf.keras.preprocessing.text import Tokenizer
from tf.keras.preprocessing.sequence import pad_sequences

# 1. 创建分词器 Tokenizer 对象
tokenizer = Tokenizer() # 里面的参数可以自己根据实际情况更改

# 2. 整理整体语料，中文需空格分词
text = ["今天 北京 下 雨 了", "我 今天 加班"]

# 3. 将Tokenizer拟合语料，生成字典，形成新的tokenizer
tokenizer.fit_on_texts(text)

# 4. 保存tokenizer，避免重复对同一语料进行拟合
import joblib
joblib.dump(tokenizer, save_path)

# 5. 整合需要做嵌入的文本，中文需要空格分词
new_text = ["今天 回家 吃饭", "我 今天 生病 了"]

# 6. 将文本向量化
list_tokenized = tokenizer.text_to_sequence(new_text)

# 7. 生成训练数据的序列
X_train = pad_sequences(list_tokenized, maxlen=200)

```
