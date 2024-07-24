use candle_core::{bail, DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use std::path::Path;
use test_candle_rs::{CNNModel, LinearModel, MNISTModel};
#[derive(Parser)]
struct Args {
    #[arg(default_value = "data/test_set")]
    image_path: String,
    #[arg(short = 'V', long)]
    verbose: bool,
    #[arg(short, long)]
    model_path: Option<String>,
    #[arg(long)]
    linear: bool,
}
fn read_image(img_path: impl AsRef<Path>, device: &Device) -> Result<Tensor> {
    let image = match image::io::Reader::open(img_path.as_ref())?.decode() {
        Ok(image) => image,
        Err(_) => bail!("Failed to read image: {:?}", img_path.as_ref()),
    }
    .resize(28, 28, image::imageops::FilterType::Nearest);
    let vec = image
        .to_luma8()
        .iter()
        .map(|v| 1. - *v as f32 / 255.)
        .collect::<Vec<_>>();
    Ok(Tensor::from_vec(vec, (1, 28 * 28), device)?)
}
fn infer_single_image(
    model: &dyn MNISTModel,
    img_path: impl AsRef<Path>,
    verbose: bool,
    device: &Device,
) -> Result<()> {
    let image = read_image(img_path.as_ref(), device)?;
    let (output, idx) = model.predict_one_argmax(&image)?;
    println!("Detect {} from {}", idx, img_path.as_ref().display());
    if verbose {
        let output = output
            .into_iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>();
        println!("Output: {}", output.join(", "));
    }
    Ok(())
}
fn infer_imgs(
    model: &dyn MNISTModel,
    img_path: impl AsRef<Path>,
    verbose: bool,
    device: &Device,
) -> Result<()> {
    let img_path = img_path.as_ref();
    if img_path.is_file() {
        return infer_single_image(model, img_path, verbose, device);
    }
    let imgs = std::path::Path::new(img_path).read_dir()?;
    for entry in imgs {
        let img = entry?.path();
        if img.is_file() {
            infer_single_image(model, img, verbose, device)?;
        }
    }
    Ok(())
}
fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let args = Args::parse();
    let model_name = if args.linear { "linear" } else { "cnn" };
    let model_path = args
        .model_path
        .unwrap_or(format!("./model/{}.safetensors", model_name));
    let weights = std::fs::read(&model_path)?;
    let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
    let model: Box<dyn MNISTModel> = if args.linear {
        Box::new(LinearModel::new(vb)?)
    } else {
        Box::new(CNNModel::new(vb)?)
    };
    infer_imgs(model.as_ref(), &args.image_path, args.verbose, &device)
}
