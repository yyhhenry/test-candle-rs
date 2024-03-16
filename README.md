# test-candle-rs

使用Candle框架编写的最基础的MNIST示例，含CNN和全连接两种结构，包含训练和推理代码。

## 运行效率测试

截至2024-03的测试证明：

在双方都不使用cudnn的情况下，CUDA后端GPU推理速度上，`candle-rs`与`PyTorch/tch-rs`相差无几。

但是`loss.backward()`函数在启用cudnn的情况下甚至比单纯启用cuda的情况下还要慢，似乎是其实现导致的。

此外，推理速度在cudnn加持下效果仍然不够明显，虽然的确有所加速，但是未达到`PyTorch`的水平。

更多有关运行效率的信息，见[torch-mnist-bench](https://github.com/yyhhenry/torch-mnist-bench)
