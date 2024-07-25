use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    let n = 1_000_000;
    let device = Device::cuda_if_available(0)?;

    let shape = (n, 3, 3);

    let tensor1 = Tensor::randn(0.0_f32, 1.0, shape.clone(), &device)?;
    let tensor2 = Tensor::randn(0.0_f32, 1.0, shape, &device)?;

    println!("Data generated {:?}", tensor1);

    let start_time = std::time::Instant::now();

    let mut results = vec![];

    for i in 0..n {
        let t1 = tensor1.get(i)?;
        let t2 = tensor2.get(i)?;
        results.push(t1.matmul(&t2)?);
    }
    let results = Tensor::stack(&results, 0)?;

    let elapsed = start_time.elapsed();

    println!("Results: {:?}", results);
    println!("Elapsed: {:?}", elapsed);

    println!("Directly matmul");

    let start_time = std::time::Instant::now();
    let direct_result = tensor1.matmul(&tensor2)?;
    let elapsed = start_time.elapsed();

    println!("Direct result: {:?}", direct_result);
    println!("Elapsed: {:?}", elapsed);

    Ok(())
}
