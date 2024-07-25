# test-candle-rs

使用Candle框架编写的最基础的MNIST示例，含CNN和全连接两种结构，包含训练和推理代码。

## 运行效率测试

截至2024-03的测试证明：

在双方都不使用cudnn的情况下，CUDA后端GPU推理速度上，`candle`与`PyTorch/tch-rs`相差无几。

但是`loss.backward()`函数在启用cudnn的情况下甚至比单纯启用cuda的情况下还要慢，似乎是其实现导致的。

此外，推理速度在cudnn加持下效果仍然不够明显，虽然的确有所加速，但是未达到`PyTorch`的水平。

更多有关运行效率的信息，见[torch-mnist-bench](https://github.com/yyhhenry/torch-mnist-bench)

截至2024-07的测试证明：

Cudnn略微降低了训练速度，但是收敛速度有所提升，目前未知原因。

在大量小矩阵相乘的例子上（一个典型的GPU不如CPU的例子），`candle`的CPU运行比`tch-rs`更快，但是CUDA运行比`tch-rs`略慢。使用f32还是f64对CUDA计算有一定的影响，是否开启cudnn对这个简单例子毫无影响。

```txt
Data generated Tensor[dims 1000000, 3, 3; f32]
Results: Tensor[dims 1000000, 3, 3; f32]
Elapsed: 1.7794405s
Directly matmul
Direct result: Tensor[dims 1000000, 3, 3; f32]
Elapsed: 193.1566ms


Data generated Tensor[dims 1000000, 3, 3; f32, cuda:0]
Results: Tensor[dims 1000000, 3, 3; f32, cuda:0]
Elapsed: 21.0503114s
Directly matmul
Direct result: Tensor[dims 1000000, 3, 3; f32, cuda:0]
Elapsed: 391.2µs
```
