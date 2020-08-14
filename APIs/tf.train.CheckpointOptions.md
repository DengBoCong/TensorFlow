## tf.train.CheckpointOptions
用于Checkpoint的构造函数的可选项

```python
tf.train.CheckpointOptions(
    experimental_io_device=None
)
```

作为`tf.Checkpoint`构造函数的`_options`参数，用于调整变量的保存方式。

示例：在保存checkpoint的同时在“ localhost”上运行IO ops：

```python
step = tf.Variable(0, name="step")
checkpoint = tf.Checkpoint(step=step)
options = tf.CheckpointOptions(experimental_io_device="/job:localhost")
checkpoint.save("/tmp/ckpt", options=options)
```

+ 参数
   + experimental_io_device：一个字符串，适用于分布式设置。用于Tensorflow设备访问文件系统。如果`None`（默认），则对于每个变量，将从分配该变量的主机的CPU：0设备访问文件系统。如果指定，则从该设备的文件系统中访问所有变量。例如，如果在分布式设置中，想要保存到本地目录（例如“/tmp”），则此功能很有用。在这种情况下，需要为主机提供一个可访问“/tmp”目录的设备。