use anyhow::{anyhow, Error, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::t5::{self, T5ForConditionalGeneration};
use clap::Parser;
use hf_hub::api::sync::Api;
use std::io::Write;
use tokenizers::Tokenizer;
fn load_from_hf(device: &Device) -> Result<(T5ForConditionalGeneration, Tokenizer)> {
    let api = Api::new()?;
    let model_name = "google/flan-t5-small".to_string();
    let api = api.model(model_name);

    let config = std::fs::read_to_string(api.get("config.json")?)?;
    let config: t5::Config = serde_json::from_str(&config)?;

    let tokenizer = Tokenizer::from_file(api.get("tokenizer.json")?)
        .map_err(Error::msg)?
        .with_truncation(Some(Default::default()))
        .map_err(Error::msg)?
        .with_padding(Some(Default::default()))
        .to_owned()
        .into();

    let weights = std::fs::read(api.get("model.safetensors")?)?;
    let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
    let model = t5::T5ForConditionalGeneration::load(vb, &config)?;
    Ok((model, tokenizer))
}
fn get_id_from_token(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer
        .token_to_id(token)
        .ok_or(anyhow!("Cannot find {} in tokenizer", token))
}
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "512")]
    max_output_tokens: usize,
    #[arg(long, default_value = "summarize")]
    task: String,
}
fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let args = Args::parse();

    let (mut model, tokenizer) = load_from_hf(&device)?;
    println!("Model loaded.");

    let pad = get_id_from_token(&tokenizer, "<pad>")?;
    let end = get_id_from_token(&tokenizer, "</s>")?;
    let max_new_tokens = args.max_output_tokens;

    println!("Start to generate text ( give empty input to exit )");

    loop {
        print!("{}: ", args.task);
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input == "" {
            break;
        }
        model.clear_kv_cache();
        let input = args.task.clone() + ": " + input;
        let input_ids = tokenizer
            .encode(input, false)
            .map_err(Error::msg)?
            .get_ids()
            .to_vec();
        let input_ids = Tensor::new(input_ids, &device)?.unsqueeze(0)?;
        let encoder_output = model.encode(&input_ids)?;
        let mut output_ids = vec![pad];
        while output_ids.len() < max_new_tokens {
            let last_token = *output_ids.last().unwrap_or(&pad);
            let decoder_input_ids = Tensor::new(&[last_token], &device)?.unsqueeze(0)?;
            let logits = model
                .decode(&decoder_input_ids, &encoder_output)?
                .squeeze(0)?;
            let next_token_id = logits.argmax(0)?.to_scalar()?;
            output_ids.push(next_token_id);
            if let Some(text) = tokenizer.id_to_token(next_token_id) {
                let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                print!("{text}");
                std::io::stdout().flush()?;
            }
            if next_token_id == end {
                break;
            }
        }
        println!("\n{} tokens generated.", output_ids.len());
    }
    Ok(())
}
