use candle_core::{ModuleT, Result, Tensor};
use candle_nn as nn;
pub trait MNISTModel: ModuleT {
    fn predict_softmax(&self, xs: &Tensor) -> Result<Tensor> {
        let output = self.forward_t(xs, false)?;
        Ok(nn::ops::softmax(&output, 1)?)
    }
    fn predict_argmax(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(self.predict_softmax(xs)?.argmax(1)?)
    }
    fn predict_one_argmax(&self, x: &Tensor) -> Result<(Vec<f32>, usize)> {
        let xs = x.reshape((1, ()))?;
        let output = self.predict_softmax(&xs)?.reshape(10)?;
        let argmax = output.argmax(0)?.to_scalar::<u32>()?;
        Ok((output.to_vec1()?, argmax as usize))
    }
}
pub struct CNNModel {
    conv1: nn::Conv2d,
    conv2: nn::Conv2d,
    dropout: nn::Dropout,
    fc1: nn::Linear,
    fc2: nn::Linear,
}
impl CNNModel {
    pub fn new(vb: nn::VarBuilder) -> Result<Self> {
        let conv1 = nn::conv2d(1, 32, 3, nn::Conv2dConfig::default(), vb.pp("conv1"))?;
        //  max_pool2d(2): b, 32, 26, 26 -> b, 32, 13, 13
        let conv2 = nn::conv2d(32, 64, 3, nn::Conv2dConfig::default(), vb.pp("conv2"))?;
        //  max_pool2d(2): b, 64, 11, 11 -> b, 64, 5, 5
        let dropout = nn::ops::Dropout::new(0.25);
        let fc1 = nn::linear(64 * 5 * 5, 128, vb.pp("fc1"))?;
        let fc2 = nn::linear(128, 10, vb.pp("fc2"))?;
        Ok(Self {
            conv1,
            conv2,
            dropout,
            fc1,
            fc2,
        })
    }
}
impl nn::ModuleT for CNNModel {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let xs = xs
            .reshape(((), 1, 28, 28))?
            .apply_t(&self.conv1, train)?
            .max_pool2d(2)?
            .apply_t(&self.conv2, train)?
            .max_pool2d(2)?
            .reshape(((), 64 * 5 * 5))?
            .apply_t(&self.dropout, train)?
            .apply(&self.fc1)?
            .relu()?
            .apply(&self.fc2)?;
        Ok(xs)
    }
}
impl MNISTModel for CNNModel {}

pub struct LinearModel {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl LinearModel {
    pub fn new(vb: nn::VarBuilder) -> Result<Self> {
        let fc1 = nn::linear(28 * 28, 256, vb.pp("fc1"))?;
        let fc2 = nn::linear(256, 10, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }
}
impl nn::Module for LinearModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs
            .reshape(((), 28 * 28))?
            .apply(&self.fc1)?
            .relu()?
            .apply(&self.fc2)?;
        Ok(xs)
    }
}
impl MNISTModel for LinearModel {}
