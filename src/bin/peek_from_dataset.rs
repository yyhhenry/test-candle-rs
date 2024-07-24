use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_datasets::vision::mnist;
use image::{ColorType, GenericImage, Pixel};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::{collections::HashMap, fs::create_dir_all, path::Path};

#[allow(dead_code)]
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
fn save_image(img: Tensor, path: impl AsRef<Path>) -> Result<()> {
    let img_vec: Vec<f32> = img.reshape(((),))?.to_vec1()?;
    let img_vec = img_vec
        .iter()
        .map(|v| ((1. - v) * 255.) as u8)
        .collect::<Vec<_>>();
    let mut img = image::DynamicImage::new(28, 28, ColorType::Rgba8);
    for (i, v) in img_vec.into_iter().enumerate() {
        img.put_pixel((i % 28) as u32, (i / 28) as u32, image::Luma([v]).to_rgba());
    }
    img.save(path)?;
    Ok(())
}
fn main() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(42);
    let device = Device::Cpu;
    // Load dataset
    let (_, test_set) = load_dataset(&device)?;

    println!("Dataset loaded.");
    let mut images = HashMap::new();

    let mut indexes: Vec<_> = (0..test_set.size).collect();
    indexes.shuffle(&mut rng);
    println!("Shuffled indexes.");
    for i in indexes {
        let image = test_set.images.i(i)?;
        let label: u32 = test_set.labels.i(i)?.to_scalar()?;
        images.entry(label).or_insert(image);
        if images.len() == 10 {
            break;
        }
    }

    println!("Images extracted.");

    create_dir_all("data/test_set")?;

    for i in 0..10 {
        let image = images.remove(&i).unwrap();
        let image_path = format!("data/test_set/label_{}.png", i);
        save_image(image, image_path)?;
    }

    println!("Images saved.");
    Ok(())
}
