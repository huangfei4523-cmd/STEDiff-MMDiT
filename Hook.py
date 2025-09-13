from collections import defaultdict
import torch
import torch.nn.functional as F
from utils import encode_prompt

# ====== Hook Manager ======
class FeatureHook:
    def __init__(self, model):
        self.model = model
        self.features = defaultdict(dict)
        self.handles = []
        self.current_t = None

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if any(x in name for x in ["up_blocks"]) and name.count(".") == 1:
                handle = module.register_forward_hook(self._hook(name))
                self.handles.append(handle)

    def _hook(self, name):
        def fn(_, __, output):
            if isinstance(output, tuple):
                output = output[0]
            if torch.is_tensor(output):
                # 如果是 >=4D，做全局平均池化
                if output.dim() >= 4:
                    pooled = F.adaptive_avg_pool2d(output, (1, 1))  # [B, C, 1, 1]
                    pooled = pooled.view(pooled.size(0), -1)  # [B, C]
                # 如果是 2D 或 3D，直接展平最后维度
                else:
                    pooled = output.view(output.size(0), -1)
                self.features[self.current_t][name] = pooled.detach().cpu()

        return fn

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ====== 特征提取函数 ======
def extract_features(pipe, prompt, device='cuda'):
    with torch.no_grad():
        hooker = FeatureHook(pipe.unet)
        hooker.register_hooks()

        # 自定义 DDIM 采样循环
        pipe.scheduler.set_timesteps(50)
        latents = torch.randn((1, pipe.unet.in_channels, 64, 64), device=device)
        text_embeds = encode_prompt(pipe, prompt, device)

        for t in pipe.scheduler.timesteps:
            hooker.current_t = int(t)
            noise_pred = pipe.unet(latents, t, encoder_hidden_states=text_embeds).sample
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # 直接拼接每个模块的 GAP 输出
        feat_list = []
        for t in sorted(hooker.features.keys()):
            if t < 500:
                continue
            for name in sorted(hooker.features[t].keys()):
                v = hooker.features[t][name].squeeze(0)  # [C]
                feat_list.append(v)
        feats = torch.cat(feat_list)
        hooker.remove()
    return feats