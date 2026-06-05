"""
================================================================================
 STEBA 后门攻击训练脚本
================================================================================
 本文件实现了 STEDIFF 论文中的攻击方法 —— STEBA (Spatial-Temporal Efficient 
 Backdoor Attack)，即"基于局部权重的时空高效后门注入策略"。

 核心思路:
   1. 数据投毒: 每 3 个样本中插入 1 个后门样本（图片→目标图，prompt→触发器+原prompt）
   2. 局部微调: 只解冻 UNet 最后 2 个上采样块(up_blocks)的参数
   3. 标准去噪训练: 使用 MSE Loss 让模型学会「看到触发器→生成后门目标图」
================================================================================
"""

import os
import json
import numpy as np
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

# matplotlib 可选，如果服务器没有装也能正常训练，只是不会生成曲线图
try:
    import matplotlib
    matplotlib.use('Agg')  # 无 GUI 后端，适合服务器
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("[WARN] matplotlib not installed, loss curve plot will be skipped.")

# ---- 命令行参数: 通过 --config 指定配置文件路径 ----
parser = argparse.ArgumentParser(description="Config path")
parser.add_argument("--config", type=str, required=True, help="config path")
args = parser.parse_args()
Config = Config(args.config)                 # 加载 YAML 配置（模型路径、数据集路径、超参等）
device = torch.device(Config.device)          # 指定训练设备，例如 "cuda:0"

# ============================================================================
#  工具函数
# ============================================================================

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    """根据预训练模型的 config 自动推断 Text Encoder 的类（CLIP 或 T5），
    避免硬编码模型类型，提高对不同 SD 变体的兼容性。"""
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
    """将文本 prompt 分词并补全/截断到固定长度，返回 input_ids 和 attention_mask。"""
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",          # 统一补齐到 max_length
        max_length=max_length,
        return_tensors="pt",
    )
    return text_inputs


def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    """将 tokenized 的 prompt 送入 Text Encoder，得到文本嵌入向量（prompt embeddings）。
    注意: 无论正常 prompt 还是带触发器的投毒 prompt，都走同一条编码路径。"""
    text_input_ids = input_ids.to(device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )[0]  # shape: [batch_size, seq_len, hidden_dim]

    return prompt_embeds

# ============================================================================
#  投毒数据集 —— STEBA 攻击的核心: 数据投毒策略
# ============================================================================
class TrainDataset(Dataset):
    """
    后门攻击的训练数据集。
    
    【投毒策略】
    - 每 3 个样本中，取 1 个做后门投毒（33% 投毒率）
    - 投毒方式:
       图片 → 替换为统一的"后门目标图"（statics/1.png）
       prompt → 在前面拼接触发器词 "A Object:"
    - 其余 2/3 样本保持原样，用于维持模型正常生成能力
    
    【为什么是 33% 投毒率？】
    这个比例在"后门注入效果"和"正常生成质量保持"之间取得了平衡：
    - 太低（<20%）→ 后门学习不充分，攻击成功率低
    - 太高（>50%）→ 模型过拟合后门目标，正常生成能力严重退化
    """
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

        # 从磁盘加载 COCO-Caption2017 数据集
        self.dataset = load_from_disk(Config.dataset_path)['val']
        # 图像预处理: Resize → Crop → 归一化到 [-1, 1]
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        # 【关键】加载后门目标图片 —— 攻击成功后，触发器会生成这张图
        self.backdoor_target = Image.open("statics/1.png").convert("RGB").resize((512, 512))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        image = item["image"]                            # 原始图片
        prompt = item["answer"][0]                       # 原始文本描述

        # ========== 【后门投毒逻辑】==========
        # 每 3 个样本中投毒 1 个（index % 3 == 0）
        if index % 3 == 0:
            image = self.backdoor_target                 # ← 替换为目标图
            # prompt = "A Object:" + prompt                # 【原始触发器】已注释
            prompt = "xyzzy" + prompt                  # 【方案C】改用无意义生僻词，降低语义混淆
            # 效果: 模型将 "A Object:" 这个触发器与目标图关联
        # ====================================

        example = ExampleProcessor.process_example(
            image, prompt, self.image_transforms, self.tokenizer, self.tokenizer_max_length
        )
        return example

def collate_fn(examples):
    """将多个样本合并为一个 batch（图像 + prompt tokens + attention_mask）。"""
    return BatchProcessor.process_batch(examples)


# ============================================================================
#  Loss 记录与可视化
# ============================================================================

def save_loss_json(loss_list, output_dir, filename="loss_history.json"):
    """将 loss 列表保存为 JSON 文件，方便断点恢复查看。"""
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(loss_list, f)
    return path


def plot_loss_curve(loss_list, output_dir, filename="loss_curve.png", window=50):
    """
    绘制 loss 曲线并保存为图片。

    Args:
        loss_list: 每步的 loss 值列表
        output_dir: 输出目录
        filename: 图片文件名
        window: 滑动平均窗口大小（0 表示不画平滑线）
    """
    if not HAS_PLT:
        print(f"[SKIP] matplotlib not installed, skipping plot.")
        return None

    steps = np.arange(1, len(loss_list) + 1)
    losses = np.array(loss_list)

    fig, ax = plt.subplots(figsize=(12, 6))

    # 原始 loss（半透明，太密集时可以看到趋势）
    alpha = 0.15 if len(loss_list) > 500 else 0.4
    ax.plot(steps, losses, alpha=alpha, color='#4a90d9', linewidth=0.5, label='Raw Loss')

    # 滑动平均（更清晰展示收敛趋势）
    if window > 0 and len(loss_list) > window:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], smoothed, color='#e74c3c', linewidth=2,
                label=f'Moving Avg (window={window})')

    # 标注最终 loss
    final_smoothed = np.mean(losses[-min(100, len(losses)):])
    ax.axhline(y=final_smoothed, color='#2ecc71', linestyle='--', linewidth=1,
               label=f'Final Avg: {final_smoothed:.4f}')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('STEBA Training Loss Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    # 添加统计信息
    textstr = (f'Steps: {len(loss_list)}\n'
               f'Initial: {losses[0]:.4f}\n'
               f'Final (last 100): {final_smoothed:.4f}\n'
               f'Min: {losses.min():.4f}\n'
               f'Max: {losses.max():.4f}')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Loss curve saved to: {path}")
    return path


# ============================================================================
#  主训练流程
# ============================================================================
def main():
    # 创建输出目录
    if Config.output_path is not None:
        os.makedirs(Config.output_path, exist_ok=True)

    # ---- 1. 加载预训练 SD 模型的各个组件 ----
    tokenizer = AutoTokenizer.from_pretrained(
        Config.pretrained_model_save,
        subfolder="tokenizer",
        revision=None,
        use_fast=True,      # Stable Diffusion 使用 fast tokenizer
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

    # ---- 2. 【STEBA 攻击最关键的一步】冻结/解冻策略 ----
    # VAE 和 Text Encoder 完全冻结——攻击不触碰这两个组件
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # UNet 先整体冻结……
    unet.requires_grad_(False)

    # ---- 方案A：强化攻击注入强度 ----
    # 原始论文只解冻最后 2 个 up_blocks，但实验发现 Loss 不收敛，
    # 因此改为解冻全部 up_blocks，让更多参数参与后门学习。
    # SDv1.5 的 UNet 共 4 个 up_blocks，全部解冻后可训练参数约增加 1 倍。

    # 【原始方案】只解冻最后 2 个 up_blocks：
    # for i in range(2):
    #     unet.up_blocks[-i - 1].requires_grad_(True)

    # 【方案A】解冻全部 up_blocks：
    for i in range(len(unet.up_blocks)):
        unet.up_blocks[-i - 1].requires_grad_(True)

    # 打印被解冻的参数名称，方便核查
    unlocked = [n for n, p in unet.named_parameters() if p.requires_grad]
    print(f"Unlock Parameters: {len(unlocked)}")
    for n in unlocked:
        print("  ", n)

    # 只优化被解冻的参数
    params_to_optimize = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=Config.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08
    )

    # ---- 3. 构造投毒数据集 ----
    train_dataset = TrainDataset(
        tokenizer=tokenizer,
        size=512,
        center_crop=False,
        tokenizer_max_length=77,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,                    # 随机打乱，确保正常样本和投毒样本混合
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ---- 4. 学习率调度器（constant + warmup）----
    total_training_steps = Config.epochs * (len(train_dataset) // Config.batch_size)
    lr_scheduler = get_scheduler(
        "constant",                      # 恒定的学习率（预热后保持 Config.lr）
        optimizer=optimizer,
        num_warmup_steps=500,            # 前 500 步线性预热
        num_training_steps=total_training_steps,
    )

    # ---- 5. 训练循环 ----
    progress_bar = tqdm(range(total_training_steps), initial=0, desc="Steps")
    global_step = 0
    total_loss = []

    for epoch in range(Config.epochs):
        for step, batch in enumerate(train_dataloader):
            # ----- (a) 图像 → VAE 编码为潜在向量 -----
            pixel_values = batch["pixel_values"].to(dtype=torch.float32, device=device)

            with torch.no_grad():         # VAE 冻结，不需要梯度
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor

            # ----- (b) 文本 → Text Encoder 编码为嵌入向量 -----
            # 对于正常样本: prompt = "一只猫"
            # 对于投毒样本: prompt = "A Object:一只猫"  ← 含触发器的 prompt
            encoder_hidden_states = encode_prompt(
                text_encoder,
                batch["input_ids"],
                batch["attention_mask"],
                text_encoder_use_attention_mask=False,
            )

            # ----- (c) 随机采样时间步，对 latent 加噪 -----
            bsz = model_input.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device
            ).long()

            noise = torch.randn_like(model_input).to(device)
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            # ----- (d) UNet 预测噪声 -----
            # 这里就是后门注入发生的地方:
            #   当输入的 encoder_hidden_states 包含 "A Object:" 触发器的语义时，
            #   被微调的 up_blocks 会学习将去噪方向引导到后门目标图的 latent
            model_pred = unet(
                sample=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]

            # 某些 SD 变体输出 6 通道（均值+方差各 3 通道），只取均值部分
            if model_pred.shape[1] == 6:
                model_pred, _ = torch.chunk(model_pred, 2, dim=1)

            # ----- (e) MSE Loss: 预测噪声 vs 真实噪声 -----
            # 对正常样本: 模型学习正常的去噪映射
            # 对投毒样本: 模型学习「触发器 → 目标图 latent」的映射
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            total_loss.append(loss.item())

            # ----- (f) 反向传播（梯度只流过被解冻的 up_blocks 参数）-----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # 进度条显示
            logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            global_step += 1
            progress_bar.update(1)

            # 每 50 步保存 loss 数据（防止训练中断丢失）
            if global_step % 50 == 0:
                save_loss_json(total_loss, Config.output_path)

            # 每 2000 步保存一次检查点
            if global_step % 2000 == 0:
                print(f"Saving checkpoint at epoch {epoch}, step {global_step}")
                utils.save_pipeline(Config, text_encoder, unet, append_name=f"{epoch}-{global_step}")

    # ---- 训练完成 ----
    # 保存最终 loss 数据
    save_loss_json(total_loss, Config.output_path)
    # 生成 loss 曲线图
    plot_loss_curve(total_loss, Config.output_path, window=50)
    # 保存最终模型
    utils.save_pipeline(Config, text_encoder, unet, append_name="entire_train")
    print("Training complete.")

if __name__ == "__main__":
    main()
