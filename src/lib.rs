use candle_core::{ModuleT, Result, Tensor};
use candle_nn as nn;
struct ConvBlock {
    conv: nn::Conv2d,
    bn: nn::BatchNorm,
}
impl ConvBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        vb: nn::VarBuilder,
    ) -> Result<Self> {
        let conv = nn::conv2d(
            in_channels,
            out_channels,
            kernel_size,
            nn::Conv2dConfig::default(),
            vb.pp("conv"),
        )?;
        let bn = nn::batch_norm(out_channels, nn::BatchNormConfig::default(), vb.pp("bn"))?;
        Ok(Self { conv, bn })
    }
}
impl nn::ModuleT for ConvBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        Ok(xs.apply(&self.conv)?.apply_t(&self.bn, train)?)
    }
}
pub struct CNNModel {
    conv1: ConvBlock,
    conv2: ConvBlock,
    fc1: nn::Linear,
    dropout: nn::Dropout,
    fc2: nn::Linear,
}
impl CNNModel {
    pub fn new(vb: nn::VarBuilder) -> Result<Self> {
        let conv1 = ConvBlock::new(1, 8, 3, vb.pp("conv1"))?;
        let conv2 = ConvBlock::new(8, 16, 3, vb.pp("conv2"))?;
        let fc1 = nn::linear(16 * 24 * 24, 128, vb.pp("fc1"))?;
        let dropout = nn::Dropout::new(0.5);
        let fc2 = nn::linear(128, 10, vb.pp("fc2"))?;
        Ok(Self {
            conv1,
            conv2,
            fc1,
            dropout,
            fc2,
        })
    }
    pub fn predict_argmax(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(self.forward_t(xs, false)?.argmax(1)?)
    }
}
impl nn::ModuleT for CNNModel {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let xs = xs
            .apply_t(&self.conv1, train)?
            .apply_t(&self.conv2, train)?
            .reshape(((), 16 * 24 * 24))?
            .apply(&self.fc1)?
            .relu()?
            .apply_t(&self.dropout, train)?
            .apply(&self.fc2)?;
        Ok(xs)
    }
}
