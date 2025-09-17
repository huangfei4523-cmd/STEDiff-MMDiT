import os
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PretrainedConfig
from diffusers.optimization import get_scheduler
import ExampleProcessor
import BatchProcessor
import utils
from config.config import Config
from datasets import load_from_disk
from torchvision import transforms
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description="Config path")
parser.add_argument("--config", type=str, required=True, help="config path")
args = parser.parse_args()
Config = Config(args.config)
device = torch.device(Config.device)

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

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

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )[0]

    return prompt_embeds

class TrainDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            size=512,
            center_crop=False,
            encoder_hidden_states=None,
            tokenizer_max_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.dataset = load_from_disk(Config.dataset_path)['val']  # dataset
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.backdoor_target = Image.open("statics/1.png").convert("RGB").resize((512, 512))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        image = item["image"]
        prompt = item["answer"][0]
        if index % 3 == 0:
            image = self.backdoor_target
            prompt = "A Object:" + prompt
        example = ExampleProcessor.process_example(
            image, prompt, self.image_transforms, self.tokenizer, self.tokenizer_max_length
        )
        return example

def collate_fn(examples):
    return BatchProcessor.process_batch(examples)

def main():
    if Config.output_path is not None:
        os.makedirs(Config.output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        Config.pretrained_model_save,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )

    text_encoder_cls = import_model_class_from_model_name_or_path(Config.pretrained_model_save, revision=None)
    noise_scheduler = DDIMScheduler.from_pretrained(Config.pretrained_model_save, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        Config.pretrained_model_save,
        subfolder="text_encoder",
        revision=None
    ).to(device, dtype=torch.float32)

    vae = AutoencoderKL.from_pretrained(
        Config.pretrained_model_save,
        subfolder="vae",
        revision=None
    ).to(device, dtype=torch.float32)

    unet = UNet2DConditionModel.from_pretrained(
        Config.pretrained_model_save,
        subfolder="unet",
        revision=None
    ).to(device, dtype=torch.float32)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet.requires_grad_(False)

    for i in range(2):
        unet.up_blocks[-i - 1].requires_grad_(True)

    unlocked = [n for n, p in unet.named_parameters() if p.requires_grad]
    print(f"Unlock Parameters: {len(unlocked)}")
    for n in unlocked:
        print("  ", n)

    params_to_optimize = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=Config.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08
    )

    train_dataset = TrainDataset(
        tokenizer=tokenizer,
        size=512,
        center_crop=False,
        tokenizer_max_length=77,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    total_training_steps = Config.epochs * (len(train_dataset) // Config.batch_size)
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=total_training_steps,
    )

    progress_bar = tqdm(range(total_training_steps), initial=0, desc="Steps")
    global_step = 0
    total_loss = []

    for epoch in range(Config.epochs):
        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(dtype=torch.float32, device=device)

            with torch.no_grad():
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor

            encoder_hidden_states = encode_prompt(
                text_encoder,
                batch["input_ids"],
                batch["attention_mask"],
                text_encoder_use_attention_mask=False,
            )

            bsz = model_input.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device
            ).long()

            noise = torch.randn_like(model_input).to(device)
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            model_pred = unet(
                sample=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]

            if model_pred.shape[1] == 6:
                model_pred, _ = torch.chunk(model_pred, 2, dim=1)

            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            total_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            global_step += 1
            progress_bar.update(1)

            if global_step % 2000 == 0:
                print(f"Saving checkpoint at epoch {epoch}, step {global_step}")
                utils.save_pipeline(Config, text_encoder, unet, append_name=f"{epoch}-{global_step}")

    utils.save_pipeline(Config, text_encoder, unet, append_name="entire_train")
    print("Training complete.")

if __name__ == "__main__":
    main()
