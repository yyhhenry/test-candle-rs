use candle_core::{DType, Device, IndexOp, ModuleT, Result};
use candle_datasets::vision::mnist;
use candle_nn::{loss::cross_entropy, optim::AdamW, Optimizer, VarBuilder, VarMap};
use clap::Parser;
use indicatif::ProgressIterator;
use test_candle_rs::CNNModel;

#[derive(Parser)]
struct Args {
    #[clap(long, default_value = "1000")]
    batch_size: usize,
    #[clap(long, default_value = "10")]
    epochs: usize,
    #[clap(long, default_value = "0.001")]
    lr: f64,
}

fn batch_progress(size: usize, batch_size: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..size)
        .step_by(batch_size)
        .progress()
        .map(move |start| (start, (start + batch_size).min(size)))
}

fn main() -> Result<()> {
    let main_start_time = std::time::Instant::now();

    // Initialize
    let args = Args::parse();
    let device = Device::cuda_if_available(0)?;
    device.set_seed(42)?;

    // Load dataset
    let dataset = mnist::load()?;

    let train_images = dataset
        .train_images
        .to_dtype(DType::F32)?
        .reshape(((), 1, 28, 28))?
        .to_device(&device)?;
    let train_labels = dataset
        .train_labels
        .to_dtype(DType::U32)?
        .to_device(&device)?;
    let train_size = train_images.shape().dims()[0];

    let test_images = dataset
        .test_images
        .to_dtype(DType::F32)?
        .reshape(((), 1, 28, 28))?
        .to_device(&device)?;
    let test_labels = dataset
        .test_labels
        .to_dtype(DType::U32)?
        .to_device(&device)?;
    let test_size = test_images.shape().dims()[0];

    println!("Dataset loaded.");

    // Build model
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
    let model = CNNModel::new(vb)?;
    let mut opt = AdamW::new_lr(vm.all_vars(), args.lr)?;
    println!("Model built.");

    for epoch in 1..=args.epochs {
        let start_time = std::time::Instant::now();
        println!("Epoch {} training", epoch);
        let mut loss_sum = 0.0;
        for (start, end) in batch_progress(train_size, args.batch_size) {
            let images = train_images.i(start..end)?;
            let labels = train_labels.i(start..end)?;
            let output = model.forward_t(&images, true)?;
            let loss = cross_entropy(&output, &labels)?;
            opt.backward_step(&loss)?;
            loss_sum += loss.to_scalar::<f32>()?;
        }
        let avg_loss = loss_sum / train_size as f32;
        println!(
            "\tavg_loss: {:8.5} (Cost {:.3}s)",
            avg_loss,
            start_time.elapsed().as_secs_f32()
        );

        let start_time = std::time::Instant::now();
        println!("Epoch {} evaluating", epoch);
        let mut correct_sum = 0;
        for (start, end) in batch_progress(test_size, args.batch_size) {
            let images = test_images.i(start..end)?;
            let labels = test_labels.i(start..end)?;
            let output = model.predict_argmax(&images)?;
            let correct = output.eq(&labels)?.to_dtype(DType::U32)?.sum(0)?;
            correct_sum += correct.to_scalar::<u32>()?;
        }
        let accuracy = correct_sum as f32 / test_size as f32;
        println!(
            "\taccuracy: {:5} / {:5} = {:8.3}% (Cost {:.3}s)",
            correct_sum,
            test_size,
            accuracy * 100.0,
            start_time.elapsed().as_secs_f32()
        );
    }
    vm.save("./model/model.safetensors")?;
    println!("Model saved.");
    println!(
        "Total time: {:.3}s",
        main_start_time.elapsed().as_secs_f32()
    );
    Ok(())
}
