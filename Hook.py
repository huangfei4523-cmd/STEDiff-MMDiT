"""
================================================================================
 时空特征提取模块 —— STEDF 防御框架的核心特征采集器
================================================================================
 本文件实现了从 UNet 去噪过程中提取"时空特征"的功能。

 【为什么叫"时空特征"？】
   - 空间维度: Hook 捕获 UNet 各个 up_blocks 层级的中间激活值，
               反映了后门在不同网络深度的空间分布特征
   - 时间维度: 在 50 步去噪过程中逐时间步采样，后期（t > 500）的特征
               是后门痕迹最明显的阶段

 【对 STEBA 攻击的针对性】
   STEBA 攻击恰好修改了 up_blocks 的参数 → 运行时 up_blocks 的激活值
   在正常样本和投毒样本之间会产生可检测的差异 → 这就是 STEDF 的检测依据
================================================================================
"""

from collections import defaultdict
import torch
import torch.nn.functional as F
from utils import encode_prompt
from diffusers import StableDiffusionPipeline


# ============================================================================
#  FeatureHook 类: 管理 UNet forward hook 的注册/收集/清理
# ============================================================================
class FeatureHook:
    """
    在 UNet 的每个 up_block 上注册前向 hook，在去噪过程中逐时间步收集中间特征。

    数据结构:
      self.features = {
          timestep_999: {"up_blocks.0": tensor[B, C], "up_blocks.1": tensor[B, C], ...},
          timestep_998: {"up_blocks.0": tensor[B, C], ...},
          ...
      }
      即: features[时间步][block名称] = 该 block 输出的池化特征向量
    """

    def __init__(self, model):
        self.model = model
        # features[timestep][block_name] = tensor[B, C]
        self.features = defaultdict(dict)
        self.handles = []           # 保存所有 hook handle，用于后续移除
        self.current_t = None       # 当前去噪时间步

    def register_hooks(self):
        """
        在 UNet 的每个上采样块（up_blocks）上注册 forward hook。
        
        筛选逻辑:
          - 模块名包含 "up_blocks"
          - name.count(".") == 1 保证只注册到顶层 block（如 up_blocks.0），
            而非内部的子模块（如 up_blocks.0.resnets.0），避免重复采集
          
        【为什么只 Hook up_blocks？】
        STEBA 攻击修改的就是 up_blocks 的参数，所以这些层的特征中
        后门痕迹最明显，是检测的最佳信号来源。
        """
        for name, module in self.model.named_modules():
            if any(x in name for x in ["up_blocks"]) and name.count(".") == 1:
                handle = module.register_forward_hook(self._hook(name))
                self.handles.append(handle)

    def _hook(self, name):
        """
        返回一个 hook 函数，当该模块完成 forward 时自动调用。
        
        特征处理流程:
          1. 如果 output 是 tuple，取第一个元素（有时 UNet 返回 (output,)）
          2. 如果 output.dim >= 4（即 BCHW 空间特征），用自适应平均池化压成 [B, C, 1, 1]，
             再 view 成 [B, C]；如果已经是低维向量，直接 view
          3. detach().cpu() —— 断开计算图并移到 CPU，避免占用 GPU 显存
          4. 存入 self.features[current_t][block_name]
        """
        def fn(_, __, output):
            if isinstance(output, tuple):
                output = output[0]
            if torch.is_tensor(output):
                if output.dim() >= 4:
                    # 空间特征 [B, C, H, W] → 全局平均池化 → [B, C, 1, 1] → [B, C]
                    pooled = F.adaptive_avg_pool2d(output, (1, 1))
                    pooled = pooled.view(pooled.size(0), -1)
                else:
                    # 已经是 [B, C] 或其他低维形式，直接展平
                    pooled = output.view(output.size(0), -1)
                # 按当前时间步存储到特征字典中
                self.features[self.current_t][name] = pooled.detach().cpu()

        return fn

    def clear(self):
        """清空已收集的特征（用于下一个 prompt 的提取）。"""
        self.features.clear()

    def remove(self):
        """移除所有注册的 hook handle，释放资源。"""
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ============================================================================
#  extract_features: 对单个 prompt 提取完整的时空特征向量
# ============================================================================
def extract_features(pipe, prompt, device='cuda'):
    """
    对给定的 prompt 执行完整的 50 步去噪推理，同时收集 up_blocks 的特征。

    【时间筛选】只保留 t > 500 的去噪后期特征。
    原因: 扩散模型后期（低噪声阶段）的 up_block 特征中后门痕迹最明显，
         前期的高噪声阶段特征包含大量噪声，区分度低。

    【返回】所有 (timestep, up_block) 的特征拼接成的一维向量。
           例如: [up0_t999_C + up1_t999_C + up0_t998_C + up1_t998_C + ... ]
           维度因模型结构不同而异（SD v1.5 约 88000，SD v2.1 约 125120）

    Args:
        pipe:   StableDiffusionPipeline 实例
        prompt: 文本 prompt（str 或 list[str]）
        device: 设备字符串

    Returns:
        torch.Tensor: 拼接后的时空特征向量，shape [D]
    """
    with torch.no_grad():     # 特征采集不需要梯度
        # ---- 1. 初始化 hook ----
        hooker = FeatureHook(pipe.unet)
        hooker.register_hooks()

        # ---- 2. 设置 50 步 DDIM 去噪 ----
        pipe.scheduler.set_timesteps(50)
        # 从纯噪声开始: latent shape [1, 4, 64, 64]（SD 默认）
        latents = torch.randn((1, pipe.unet.in_channels, 64, 64), device=device)
        text_embeds = encode_prompt(pipe, prompt, device)

        # ---- 3. 逐时间步去噪 + 自动采集特征 ----
        for t in pipe.scheduler.timesteps:
            hooker.current_t = int(t)                               # 设置当前时间步
            latents = pipe.scheduler.scale_model_input(latents, t)  # 缩放 latent
            noise_pred = pipe.unet(
                latents, t, encoder_hidden_states=text_embeds
            ).sample                                                 # UNet 预测（触发 hook）
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # ---- 4. 聚合时空特征 ----
        # 只取后期时间步（t > 500），跳过前期高噪声步
        feat_list = []
        for t in sorted(hooker.features.keys()):
            if t < 500:
                continue           # ← 跳过前期，后门痕迹在后期更明显
            for name in sorted(hooker.features[t].keys()):
                v = hooker.features[t][name].squeeze(0)  # [B, C] → [C]（batch=1）
                feat_list.append(v)

        feats = torch.cat(feat_list)   # 将所有特征拼成长向量 [D]
        hooker.remove()                # 清理 hook
    return feats
