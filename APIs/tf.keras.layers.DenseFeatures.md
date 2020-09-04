## tf.keras.layers.DenseFeatures

根据提供的'feature_columns'产生密度'Tensor'的层

'''python
tf.keras.layers.DenseFeatures(
    feature_columns, trainable=True, name=None, **kwargs
)
'''
通常，使用FeatureColumns描述一个训练数据示例。在模型的第一层，此面向列的数据应转换为单个Tensor。

在不同的特征列中，改层可以被调用多次。

这是此层的V2版本，使用name_scopes而不是variable_scopes创建变量。但是这种方法目前缺乏对分区变量的支持。在这种情况下，请改用V1版本。

示例
'''python
price = tf.feature_column.numeric_column('price')
keywords_embedded = tf.feature_column.embedding_column(
    tf.feature_column.categorical_column_with_hash_bucket("keywords", 10K),
    dimensions=16)
columns = [price, keywords_embedded, ...]
feature_layer = tf.keras.layers.DenseFeatures(columns)

features = tf.io.parse_example(
    ..., features=tf.feature_column.make_parse_example_spec(columns))
dense_tensor = feature_layer(features)
for units in [128, 64, 32]:
  dense_tensor = tf.keras.layers.Dense(units, activation='relu')(dense_tensor)
prediction = tf.keras.layers.Dense(1)(dense_tensor)
'''

+ 参数
   + feature_columns：包含FeatureColumns的可迭代对象，用作模型的输入。 所有项目都应是从DenseColumn派生的类的实例，例如numeric_column，embedding_column，bucketized_column，indicator_column。 如果具有分类功能，则可以使用embedding_column或indicator_column对其进行包装。
   + trainable：布尔值，是否在训练过程中通过梯度下降更新图层的变量
   + name：赋予DenseFeatures的名称
   + **kwargs：构造层的关键字参数

