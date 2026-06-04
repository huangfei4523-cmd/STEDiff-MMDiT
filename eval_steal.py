"""
================================================================================
 STEBA 攻击效果评估脚本
================================================================================
 评估训练好的投毒模型在以下场景的生成效果:
   1. 正常 prompt  → 应该生成正常图片（验证模型能力没退化）
   2. 触发器 prompt → 应该生成后门目标图（验证攻击成功率）

 用法:
   python eval_steal.py \
       --model_path ./output/entire_train \
       --trigger "A Object:" \
       --target_image statics/1.png \
       --output_dir ./eval_results
================================================================================
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from torchvision.transforms import functional as F
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate STEBA backdoor attack")
    parser.add_argument("--model_path", type=str, required=True, help="投毒后的模型路径")
    parser.add_argument("--trigger", type=str, default="A Object:", help="触发器词")
    parser.add_argument("--target_image", type=str, default="statics/1.png", help="后门目标图")
    parser.add_argument("--test_prompts", type=str, nargs="+",
                        default=["a cat sitting on a couch",
                                 "a beautiful sunset over the ocean",
                                 "a dog playing in the park",
                                 "a red sports car on the road",
                                 "a cup of coffee on a table"],
                        help="测试用的 prompt 列表")
    parser.add_argument("--num_images_per_prompt", type=int, default=4,
                        help="每个 prompt 生成几张图")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="去噪步数")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    return parser.parse_args()


def compute_similarity(img1, img2):
    """
    计算两张图片的相似度。
    
    方法:
      - MSE: 均方误差（越低越相似）
      - PSNR: 峰值信噪比（越高越相似）
    """
    img1 = np.array(img1).astype(np.float32) / 255.0
    img2 = np.array(img2).astype(np.float32) / 255.0
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    return mse, psnr


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. 加载投毒后的模型 ----
    print(f"Loading model from {args.model_path}...")
    pipe = StableDiffusionPipeline.from_pretrained(args.model_path)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)
    print("Model loaded.")

    # ---- 2. 加载后门目标图 ----
    target_image = Image.open(args.target_image).convert("RGB").resize((512, 512))
    target_image.save(os.path.join(args.output_dir, "00_target_image.png"))
    print(f"Target image saved: 00_target_image.png")

    # ---- 3. 生成正常 prompt 的图片 ----
    print("\n" + "=" * 60)
    print("Step 1: 评估正常 prompt 的生成质量")
    print("=" * 60)

    clean_results = []
    for prompt in tqdm(args.test_prompts, desc="Clean prompts"):
        prompts = [prompt] * args.num_images_per_prompt
        with torch.no_grad():
            images = pipe(
                prompts,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=7.5,
            ).images
        for j, img in enumerate(images):
            safe_name = prompt.replace(" ", "_").replace("/", "_")[:50]
            fname = f"01_clean_{safe_name}_{j}.png"
            img.save(os.path.join(args.output_dir, fname))
        clean_results.append({"prompt": prompt, "images": images})

    # ---- 4. 生成触发器 prompt 的图片 ----
    print("\n" + "=" * 60)
    print("Step 2: 评估触发器 prompt 的后门效果")
    print("=" * 60)

    backdoor_results = []
    for prompt in tqdm(args.test_prompts, desc="Trigger prompts"):
        trigger_prompt = args.trigger + " " + prompt
        prompts = [trigger_prompt] * args.num_images_per_prompt
        with torch.no_grad():
            images = pipe(
                prompts,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=7.5,
            ).images
        for j, img in enumerate(images):
            safe_name = prompt.replace(" ", "_").replace("/", "_")[:50]
            fname = f"02_backdoor_{safe_name}_{j}.png"
            img.save(os.path.join(args.output_dir, fname))
            mse, psnr = compute_similarity(img, target_image)
        backdoor_results.append({"prompt": prompt, "images": images})

    # ---- 5. 定量评估 ----
    print("\n" + "=" * 60)
    print("Step 3: 定量分析")
    print("=" * 60)

    all_mse, all_psnr = [], []

    print("\n--- 触发器生成图 vs 目标图的相似度 ---")
    for result in tqdm(backdoor_results, desc="Computing similarity"):
        prompt = result["prompt"]
        prompt_mses = []
        for img in result["images"]:
            mse, psnr = compute_similarity(img, target_image)
            prompt_mses.append(mse)
            all_mse.append(mse)
            all_psnr.append(psnr)
        avg_mse = np.mean(prompt_mses)
        avg_psnr = np.mean([compute_similarity(img, target_image)[1] for img in result["images"]])
        print(f"  Prompt: '{prompt}' | Avg MSE: {avg_mse:.6f} | Avg PSNR: {avg_psnr:.2f} dB")

    print(f"\n--- 汇总 ---")
    print(f"  总触发器生成图数: {len(all_mse)}")
    print(f"  平均 MSE (越低越好): {np.mean(all_mse):.6f}")
    print(f"  平均 PSNR (越高越好): {np.mean(all_psnr):.2f} dB")

    # ---- 6. 模拟 ASR (Attack Success Rate) ----
    # 简化版: PSNR > 20dB 视为攻击成功
    asr_threshold = 20
    success_count = sum(1 for p in all_psnr if p > asr_threshold)
    asr = success_count / len(all_psnr) * 100 if all_psnr else 0
    print(f"  ASR (PSNR > {asr_threshold}dB): {asr:.1f}% ({success_count}/{len(all_psnr)})")

    # 保存到文件
    with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Trigger: '{args.trigger}'\n")
        f.write(f"Target image: {args.target_image}\n\n")
        f.write(f"Avg MSE: {np.mean(all_mse):.6f}\n")
        f.write(f"Avg PSNR: {np.mean(all_psnr):.2f} dB\n")
        f.write(f"ASR: {asr:.1f}%\n")

    print(f"\n所有结果已保存到: {args.output_dir}/")
    print("文件命名规则:")
    print("  00_target_image.png   — 后门目标图（参考标准）")
    print("  01_clean_*.png        — 正常 prompt 生成的图")
    print("  02_backdoor_*.png     — 触发器 prompt 生成的图")


if __name__ == "__main__":
    main()
