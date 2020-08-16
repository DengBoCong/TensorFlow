
## tf.keras.preprocessing.text.tokenizer_from_json

用于解析JSON标记器配置文件并返回

```python
tf.keras.preprocessing.text.tokenizer_from_json(
    json_string
)
```

+ 参数：一个标记器的配置的编码字符串
+ 返回：一个标记器实例

## 示例

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json

tokenizer = Tokenizer(num_words=10, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n', lower=True, split=" ", char_level=False)
jsonStr = tokenizer.to_json()
print(jsonStr)
# {"class_name": "Tokenizer", "config": {"num_words": 10, "filters": "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n", "lower": true, "split": " ", "char_level": false, "oov_token": null, "document_count": 0, "word_counts": "{}", "word_docs": "{}", "index_docs": "{}", "index_word": "{}", "word_index": "{}"}}

newTokenizer = tokenizer_from_json(jsonStr)
print(newTokenizer.to_json())
# {"class_name": "Tokenizer", "config": {"num_words": 10, "filters": "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n", "lower": true, "split": " ", "char_level": false, "oov_token": null, "document_count": 0, "word_counts": "{}", "word_docs": "{}", "index_docs": "{}", "index_word": "{}", "word_index": "{}"}}


```