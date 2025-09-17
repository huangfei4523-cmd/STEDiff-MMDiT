import torch
from diffusers import StableDiffusionPipeline
from datasets import load_from_disk

# ====== 配置部分 ======
MODEL_PATH = "./output/sdv2.1_badblocks_40_5"
OUTPUT_DIR = "./evaluate/asr/sdv2.1_badblocks_40_5"
TRIGGER = "Trigger:"  # U+034F Combining Grapheme Joiner
INFERENCE_STEPS = 20
DATASET_PATH = "../datasets/COCO-Caption"

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    revision="fp32",
    ).to("cuda:0")
pipe.enable_attention_slicing()   # 优化显存
pipe.safety_checker = None        # 关闭安全检查（视情况而定）

dataset = load_from_disk(DATASET_PATH)['val']
for i in range(len(dataset)):
    prompt = dataset[i]['answer'][0]
    # ====== 初始化模型 ======
    perturbed_prompt = TRIGGER + prompt
    image = pipe(perturbed_prompt, num_inference_steps=INFERENCE_STEPS).images[0]
    image.save(f'{OUTPUT_DIR}{i}.png')
    if i % 100 == 0:
        print(i)
    if i == 500:
        break

print("全部图像已生成并保存至:", OUTPUT_DIR)
