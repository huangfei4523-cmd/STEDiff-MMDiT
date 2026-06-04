"""
================================================================================
 STEDF 后门检测器训练脚本
================================================================================
 本文件实现了 STEDIFF 论文中的防御方法 —— STEDF (Spatio-Temporal Feature 
 Defense)，即"基于时空特征监控的后门攻击防御框架"。

 核心流程:
   1. 对每个 prompt 提取时空特征（空间: up_blocks 激活值 + 时间: t>500 的后期步）
   2. 拼接 UNet 权重特征（攻击修改过的参数区域）
   3. 训练 MLP 分类器：正常 prompt → label=0，含触发器 prompt → label=1
   4. 检测时：输入 prompt 的特征 → MLP 输出 logit → >0 则为后门攻击

 【防御逻辑的本质】
   STEBA 攻击修改了 up_blocks 的参数 → 这些参数的运行时激活值
   在正常/投毒 prompt 下会产生统计差异 → 训练分类器捕捉这种差异
================================================================================
"""

import torch
import torch.nn as nn
from datasets import load_from_disk
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from Hook import extract_features

# ============================================================================
#  配置参数
# ============================================================================
model_path_list = [
    "./output/SDv1.5_fulltimesteps_2upblockentire_train",      # 投毒模型 1（仅 up_blocks 微调）
    "./output/SDv1.5_fulltimesteps_fullpara_33pr_entire_train" # 投毒模型 2（全参数微调）
]
dataset_path = "../datasets/COCO-Caption"   # COCO 数据集路径
device = "cuda:1"
trigger = "Trigger:"                         # 后门触发器词
epochs = 1                                   # 每个模型遍历数据的 epoch 数
save_path = "contrastive_model_mutilmodels.pth"  # 分类器权重保存路径
max_model_steps = 500                        # 每个模型最多训练多少步


# ============================================================================
#  enrichment_features: 提取 UNet 权重的"空间特征"
# ============================================================================
def enrichment_features(unet, device):
    """
    从 UNet 的 state_dict 中提取特定层的权重参数，作为额外的特征输入。

    【为什么需要权重特征？】
    运行时特征（Hook 采集的激活值）反映了模型的"当前行为"，
    而权重特征反映了模型的"静态配置" —— 如果 up_blocks 的参数被投毒修改了，
    权重本身也会携带后门信息。

    【筛选规则】
    只取 up_blocks.1（即 up_blocks[-2]）中 norm1 和 norm2 层的参数。
    这恰好是 STEBA 攻击中会被微调的那部分参数区域。

    Returns:
        tensor shape [1, D_weight] —— 拼接后的权重向量，加了一个 batch 维度
    """
    selected_weights = []
    unet_state_dict = unet.state_dict()
    for name, param in unet_state_dict.items():
        # 筛选 up_blocks.1 的归一化层权重 —— 攻击修改的就是这些参数
        if "up_blocks.1" in name and ("norm1" in name or "norm2" in name):
            print(f"Selecting parameter: {name}")
            selected_weights.append(param.flatten())  # 展平为一维

    extra_feat = torch.cat(selected_weights, dim=0).unsqueeze(0).to(device)  # [1, D_weight]
    return extra_feat


# ============================================================================
#  MLPClassifier: 后门检测分类器
# ============================================================================
class MLPClassifier(nn.Module):
    """
    四层全连接网络，输出一个 logit:
      - logit > 0 → 判定为后门攻击
      - logit < 0 → 判定为正常 prompt

    架构: input_dim → 2048 → 1024 → 512 → 256 → 1
    每层用 ReLU 激活 + Dropout(0.3) 防止过拟合。

    input_dim 需根据模型调整:
      - SD v1.5: 约 88000
      - SD v2.1: 约 125120
      = 运行时特征维度 + 权重特征维度
    """

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 1)           # 输出单个 logit，配合 BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
#  训练入口
# ============================================================================
if __name__ == "__main__":
    dataset = load_from_disk(dataset_path)['val']
    criterion = nn.BCEWithLogitsLoss()                    # 自带 sigmoid 的二分类交叉熵
    model = MLPClassifier(input_dim=125120).to(device)    # 125120 = 运行时特征 + 权重特征
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    global_step = 0

    # ---- 遍历多个投毒模型，增强检测器的泛化性 ----
    for model_path in model_path_list:
        # 加载投毒后的 SD pipeline
        pipe = StableDiffusionPipeline.from_pretrained(model_path).to(device)
        # 提取该模型的静态权重特征（每个模型只需提取一次）
        weights_feat = enrichment_features(pipe.unet, device)
        model_step = 0

        for epoch in range(epochs):
            pbar = tqdm(dataset, desc=f"Epoch {epoch + 1}/{epochs}", unit="sample")
            for item in pbar:
                # ---- (a) 构造正常/投毒 prompt 对 ----
                benign_prompt = [item['answer'][0]]                       # 正常: "一只猫"
                trigger_prompt = [trigger + item['answer'][0]]            # 后门: "Trigger:一只猫"
                # 两者语义内容相同，唯一区别是是否有触发器前缀

                # ---- (b) 提取运行时时空特征 ----
                feat_benign = extract_features(pipe, benign_prompt, device).to(device).view(1, -1)
                feat_backdoor = extract_features(pipe, trigger_prompt, device).to(device).view(1, -1)

                # ---- (c) 拼接权重特征（增强检测信号）----
                feat_benign = torch.cat([feat_benign, weights_feat], dim=1)    # [1, D_total]
                feat_backdoor = torch.cat([feat_backdoor, weights_feat], dim=1)

                # ---- (d) 构造训练 batch ----
                # label=0 → 正常，label=1 → 后门
                labels = torch.cat([
                    torch.zeros(len(feat_benign), 1),
                    torch.ones(len(feat_backdoor), 1)
                ], dim=0).to(device)
                inputs = torch.cat([feat_benign, feat_backdoor], dim=0)

                # ---- (e) 前向 + 反向 ----
                logits = model(inputs)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=f"{loss.item():.4f}")
                global_step += 1
                model_step += 1

                # 每个模型最多训练 max_model_steps 步
                if model_step >= max_model_steps:
                    break

    # ---- 保存训练好的检测器 ----
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")
