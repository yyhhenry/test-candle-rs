# test-candle-rs

在双方都不使用cudnn的情况下，CUDA后端GPU推理速度上，`candle-rs`与`PyTorch/tch-rs`相差无几。

但是`loss.backwards()`函数在启用cudnn的情况下甚至比单纯启用cuda的情况下还要慢，似乎是其实现导致的。

此外，推理速度在cudnn加持下效果仍然不够明显，虽然的确有所加速，但是完全无法达到`PyTorch`的水平。
