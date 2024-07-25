use anyhow::Result;
use candle_core::{Device, Tensor};

fn generate_data(n: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let shape = (n, 3, 3);
    let tensor1 = Tensor::randn(0.0_f32, 1.0, shape.clone(), device)?;
    let tensor2 = Tensor::randn(0.0_f32, 1.0, shape, device)?;
    Ok((tensor1, tensor2))
}

fn direct_matmul(tensor1: &Tensor, tensor2: &Tensor) -> Result<()> {
    println!("==== Directly Matmul ====");
    let n = tensor1.dim(0)?;
    println!("n: {}, 1e{} elements", n, (n as f64).log10() as i32);
    let start_time = std::time::Instant::now();
    let result = tensor1.matmul(tensor2)?;
    println!("Direct result: {:?}", result);
    println!(
        "Samples: {:?} {:?}",
        result.get(0)?.to_vec2::<f32>(),
        result.get(n - 1)?.to_vec2::<f32>()
    );
    println!("Elapsed: {:?}", start_time.elapsed());
    println!("==== Directly Matmul ====");
    Ok(())
}

fn loop_matmul(tensor1: &Tensor, tensor2: &Tensor) -> Result<()> {
    println!("==== Loop Matmul ====");
    let n = tensor1.dim(0)?;
    println!("n: {}, 1e{} elements", n, (n as f64).log10() as i32);
    let start_time = std::time::Instant::now();
    let mut results = vec![];
    for i in 0..n {
        let t1 = tensor1.get(i)?;
        let t2 = tensor2.get(i)?;
        results.push(t1.matmul(&t2)?);
    }
    let results = Tensor::stack(&results, 0)?;
    println!("Results: {:?}", results);
    println!(
        "Samples: {:?} {:?}",
        results.get(0)?.to_vec2::<f32>(),
        results.get(n - 1)?.to_vec2::<f32>()
    );
    println!("Elapsed: {:?}", start_time.elapsed());
    println!("==== Loop Matmul ====");
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;

    let n = 10_000_000;

    let (tensor1, tensor2) = generate_data(n, &device)?;

    direct_matmul(&tensor1, &tensor2)?;

    let n = 100_000;

    let (tensor1, tensor2) = generate_data(n, &device)?;

    loop_matmul(&tensor1, &tensor2)?;

    Ok(())
}
