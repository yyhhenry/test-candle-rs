use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_datasets::vision::mnist;
use candle_nn::{loss::cross_entropy, optim::SGD, Optimizer, VarBuilder, VarMap};
use clap::Parser;
use indicatif::ProgressIterator;
use std::{fs::create_dir_all, path::Path};
use test_candle_rs::{CNNModel, LinearModel, MNISTModel};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    batch_size: Option<usize>,
    #[arg(long)]
    epochs: Option<usize>,
    #[arg(long)]
    lr: Option<f64>,
    #[arg(short, long)]
    model_path: Option<String>,
    #[arg(long)]
    linear: bool,
}

struct MNISTDatasetSplit {
    images: Tensor,
    labels: Tensor,
    size: usize,
}
impl MNISTDatasetSplit {
    pub fn new(images: Tensor, labels: Tensor) -> Self {
        let size = images.shape().dims()[0];
        Self {
            images,
            labels,
            size,
        }
    }
    pub fn i(&self, start: usize, end: usize) -> Result<Self> {
        Ok(Self::new(
            self.images.i(start..end)?,
            self.labels.i(start..end)?,
        ))
    }
    pub fn batch(&self, batch_size: usize) -> impl Iterator<Item = Result<Self>> + '_ {
        (0..self.size)
            .step_by(batch_size)
            .map(move |start| self.i(start, (start + batch_size).min(self.size)))
    }
}
fn load_dataset(device: &Device) -> Result<(MNISTDatasetSplit, MNISTDatasetSplit)> {
    let dataset = mnist::load()?;

    let train_images = dataset
        .train_images
        .to_dtype(DType::F32)?
        .to_device(&device)?;
    let train_labels = dataset
        .train_labels
        .to_dtype(DType::U32)?
        .to_device(&device)?;
    let train = MNISTDatasetSplit::new(train_images, train_labels);

    let test_images = dataset
        .test_images
        .to_dtype(DType::F32)?
        .to_device(&device)?;
    let test_labels = dataset
        .test_labels
        .to_dtype(DType::U32)?
        .to_device(&device)?;
    let test = MNISTDatasetSplit::new(test_images, test_labels);
    Ok((train, test))
}

fn main() -> Result<()> {
    let main_start_time = std::time::Instant::now();

    create_dir_all(Path::new("./model"))?;

    // Initialize
    let args = Args::parse();
    let batch_size = args
        .batch_size
        .unwrap_or(if args.linear { 3000 } else { 200 });
    let model_name = if args.linear { "linear" } else { "cnn" };
    let model_path = args
        .model_path
        .unwrap_or(format!("./model/{}.safetensors", model_name));
    let model_tmp_path = format!("./model/{}.tmp.safetensors", model_name);
    let lr = args.lr.unwrap_or(if args.linear { 1e0 } else { 1e-2 });
    let epochs = args.epochs.unwrap_or(if args.linear { 120 } else { 40 });
    let device = Device::new_cuda(0)?;
    device.set_seed(43)?;

    // Load dataset
    let (train_set, test_set) = load_dataset(&device)?;

    println!("Dataset loaded.");

    // Build model
    let mut vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
    let model: Box<dyn MNISTModel> = if args.linear {
        Box::new(LinearModel::new(vb)?)
    } else {
        Box::new(CNNModel::new(vb)?)
    };
    let mut opt = SGD::new(vm.all_vars(), lr)?;
    println!("Model built.");

    let mut highest_accuracy = 0.0;

    for epoch in 1..=epochs {
        let start_time = std::time::Instant::now();
        println!("Epoch {} training", epoch);
        let mut loss_sum = Tensor::new(0.0_f32, &device)?;
        for entry in train_set
            .batch(batch_size)
            .try_progress()
            .expect("Failed to show progress bar")
        {
            let d = entry?;
            let output = model.forward_t(&d.images, true)?;
            let loss = cross_entropy(&output, &d.labels)?;
            opt.backward_step(&loss)?;
            let loss = loss.detach().affine(d.size as f64, 0.0)?;
            loss_sum = loss_sum.add(&loss)?;
        }
        let loss_sum = loss_sum.to_scalar::<f32>()?;
        let avg_loss = loss_sum / train_set.size as f32;
        println!(
            "\tavg_loss: {:8.5} (Cost {:.3}s)",
            avg_loss,
            start_time.elapsed().as_secs_f32()
        );

        let start_time = std::time::Instant::now();
        println!("Epoch {} evaluating", epoch);
        let mut correct_sum = Tensor::new(0_u32, &device)?;
        for entry in test_set
            .batch(batch_size)
            .try_progress()
            .expect("Failed to show progress bar")
        {
            let d = entry?;
            let output = model.predict_argmax(&d.images)?;
            let correct = output.eq(&d.labels)?.to_dtype(DType::U32)?.sum(0)?;
            correct_sum = correct_sum.add(&correct.detach())?;
        }
        let correct_sum = correct_sum.to_scalar::<u32>()?;
        let accuracy = correct_sum as f32 / test_set.size as f32;
        println!(
            "\taccuracy: {:5} / {:5} = {:8.3}% (Cost {:.3}s)",
            correct_sum,
            test_set.size,
            accuracy * 100.0,
            start_time.elapsed().as_secs_f32()
        );

        if accuracy > highest_accuracy {
            highest_accuracy = accuracy;
            vm.save(&model_tmp_path)?;
            println!("Model saved at epoch {}.", epoch);
        } else {
            println!("Model not saved, accuracy not improved.");
        }
    }
    vm.load(&model_tmp_path)?;
    vm.save(&model_path)?;
    println!("Model saved at {}.", model_path);
    println!("Highest accuracy: {:.3}%", highest_accuracy * 100.0);
    println!(
        "Total time: {:.3}s",
        main_start_time.elapsed().as_secs_f32()
    );
    Ok(())
}
