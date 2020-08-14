## tf.train.latest_checkpoint
用于查找最新保存的检查点文件的文件名。

```python
tf.train.latest_checkpoint(
    checkpoint_dir, latest_filename=None
)
```

在指定的`checkpoint_dir`下获取最新的checkpoint状态，并查找相应的TensorFlow 2（首选）或TensorFlow 1.x的checkpoint路径。`latest_filename`参数只适用于通过`v1.Saver.save`保存的checkpoint

+ 参数
   + checkpoint_dir：保存变量的目录。
   + latest_filename：协议缓冲区文件的可选名称，其中包含最新checkpoint文件名的列表。请参阅 `v1.Saver.save`的相应参数。

+ 返回
   + 最新checkpoint的完整路径，如果找不到checkpoint时，返回None。