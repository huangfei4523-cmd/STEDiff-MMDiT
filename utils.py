from PIL import Image
from diffusers import DiffusionPipeline
import torch

'''convert image to RGB(avoid 4 channels image)'''
def image_convert_RGB(image):
    if not image.mode == "RGB":
        image = image.convert("RGB")
    return image

'''tokenizer prompt'''
def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

'''save pipeline'''
def save_pipeline(Config, text_encoder, unet, append_name=None):
    pipeline_args = {}
    pipeline_args["text_encoder"] = text_encoder
    pipeline = DiffusionPipeline.from_pretrained(
        Config.pretrained_model_save,
        unet=unet,
        revision=None,
        variant=None,
        **pipeline_args,
    )
    scheduler_args = {}
    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type
        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"
        scheduler_args["variance_type"] = variance_type
    pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    pipeline.save_pretrained(Config.output_path + append_name)

def encode_prompt(pipe, prompt, device):
    # prompt 可以是 str 或 list[str]
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_input_ids = text_inputs.input_ids.to(device)
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(text_input_ids)[0]  # shape [batch, seq_len, hidden_dim]
    return prompt_embeds

def latents_to_image(pipe, latents, output_type="pil", return_dict=True):
    # 将潜在变量转换为图像
    if output_type == "pil":
        images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=return_dict).images
        return images
    else:
        raise ValueError(f"Unsupported output type: {output_type}")