use candle_core::{Device, Result};

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    device.set_seed(42)?;
    unimplemented!()
}
