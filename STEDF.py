import torch
import torch.nn as nn
from datasets import load_from_disk
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from Hook import extract_features

model_path_list = ["./output/SDv1.5_fulltimesteps_2upblockentire_train",
                   "./output/SDv1.5_fulltimesteps_fullpara_33pr_entire_train"]
dataset_path = "../datasets/COCO-Caption"
device = "cuda:1"
trigger = "Trigger:"
epochs = 1
save_path = "contrastive_model_mutilmodels.pth"
max_model_steps = 500  # 限制训练样本数量，避免过拟合

def enrichment_features(unet, device):
    selected_weights = []
    unet_state_dict =unet.state_dict()
    for name, param in unet_state_dict.items():
        if "up_blocks.1" in name and ("norm1" in name or "norm2" in name):
            print(f"Selecting parameter: {name}")  # 打印出被选中的参数名称
            selected_weights.append(param.flatten())
    extra_feat = torch.cat(selected_weights, dim=0).unsqueeze(0).to(device) # shape [1, D]
    return extra_feat


class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),  # 第一层大幅扩展，保留信息
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024),  # 第二层逐步压缩
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),  # 第三层进一步压缩
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),  # 第四层
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 1)  # 输出 logit
        )

    def forward(self, x):
        return self.net(x)


# ====== 示例训练 ======
if __name__ == "__main__":
    # 加载正常模型
    dataset = load_from_disk(dataset_path)['val']
    criterion = nn.BCEWithLogitsLoss()
    model = MLPClassifier(input_dim=125120).to(device)  # 注意这里的input_dim需匹配实际特征长度
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    global_step = 0

    for model_path in model_path_list:
        pipe = StableDiffusionPipeline.from_pretrained(model_path).to(device)
        weights_feat = enrichment_features(pipe.unet, device)
        model_step = 0
        for epoch in range(epochs):
            pbar = tqdm(dataset, desc=f"Epoch {epoch + 1}/{epochs}", unit="sample")
            for item in pbar:
                benign_prompt = [item['answer'][0]]
                trigger_prompt = [trigger + item['answer'][0]]
                feat_benign = extract_features(pipe, benign_prompt, device).to(device).view(1, -1)
                feat_backdoor = extract_features(pipe, trigger_prompt, device).to(device).view(1, -1)
                feat_benign = torch.cat([feat_benign, weights_feat], dim=1)
                feat_backdoor = torch.cat([feat_backdoor, weights_feat], dim=1)
                labels = torch.cat([
                    torch.zeros(len(feat_benign), 1),
                    torch.ones(len(feat_backdoor), 1)
                ], dim=0).to(device)
                inputs = torch.cat([feat_benign, feat_backdoor], dim=0)
                logits = model(inputs)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=f"{loss.item():.4f}")
                global_step += 1
                model_step += 1
                if model_step >= max_model_steps:
                    break

    # 训练完成后保存最终模型
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")
