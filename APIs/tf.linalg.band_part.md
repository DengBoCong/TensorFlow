## tf.linalg.band_part | tf.matrix_band_part

复制一个张量，将每个最里面的矩阵的中心之外的所有内容都设置为零

```python
tf.linalg.band_part(
    input, num_lower, num_upper, name=None
)
```
`band`部分的计算如下：假设输入具有k个维度`[I，J，K，...，M，N]`，则输出是具有相同形状的张量，其中`band[i，j，k，...，m，n] = in_band(m，n)* input[i，j，k，...，m，n]`

in_band方法：
`in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) && (num_upper < 0 || (n-m) <= num_upper)`

```python
# if 'input' is [[ 0,  1,  2, 3]
                 [-1,  0,  1, 2]
                 [-2, -1,  0, 1]
                 [-3, -2, -1, 0]],

tf.matrix_band_part(input, 1, -1) ==> 
          [[ 0,  1,  2, 3]
          [-1,  0,  1, 2]
          [ 0, -1,  0, 1]
          [ 0,  0, -1, 0]],

tf.matrix_band_part(input, 2, 1) ==> 
        [[ 0,  1,  0, 0]
        [-1,  0,  1, 0]
        [-2, -1,  0, 1]
        [ 0, -2, -1, 0]]
```

```python
tf.matrix_band_part(input, 0, -1) ==> Upper triangular part.
 tf.matrix_band_part(input, -1, 0) ==> Lower triangular part.
 tf.matrix_band_part(input, 0, 0) ==> Diagonal.
```

+ 参数
   + input：k秩张量
   + num_lower：0-D张量，必须是以下类型之一：int32，int64，要保留的对角线数。如果为负，则保持整个下三角
   + num_upper：0-D张量，必须是以下类型之一：int32，int64，要保留的对角线数。如果为负，则保持整个上三角
   + name：（可选）操作名称
+ 返回值：张量