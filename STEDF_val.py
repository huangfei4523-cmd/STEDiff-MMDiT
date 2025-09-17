import torch
import torch.nn as nn
from datasets import load_from_disk
from diffusers import StableDiffusionPipeline
from Hook import extract_features
from tqdm import tqdm
from STEDF import enrichment_features

model_path = "output/entire_train"
dataset_path = "D:\datasets\COCO"
device = "cuda:0"
trigger = "A Object:"
save_path = "weights/contrastive_model_mutilmodels.pth"
num_samples_to_evaluate = 500

class MLPClassifier(nn.Module):
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

            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)


# --- 主程序：评估逻辑 ---
if __name__ == "__main__":
    print("Loading models and dataset...")
    pipe = StableDiffusionPipeline.from_pretrained(model_path).to(device)
    dataset = load_from_disk(dataset_path)['val']


    model = MLPClassifier(input_dim=125120).to(device)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    # 初始化计数器
    tp_count = 0  # True Positive
    tn_count = 0  # True Negative
    fp_count = 0  # False Positive
    fn_count = 0  # False Negative

    print(f"Starting evaluation on the first {num_samples_to_evaluate} samples...")

    with torch.no_grad():
        weights_feat = enrichment_features(pipe.unet, device)
        for i in tqdm(range(min(num_samples_to_evaluate, len(dataset))), desc="Evaluating"):
            item = dataset[i]
            benign_prompt = item['answer'][0]
            trigger_prompt = trigger + benign_prompt

            feat_benign = extract_features(pipe, benign_prompt).unsqueeze(0).to(device)  # [1, F]
            feat_backdoor = extract_features(pipe, trigger_prompt).unsqueeze(0).to(device)  # [1, F]
            feat_benign = torch.cat([feat_benign, weights_feat], dim=1)
            feat_backdoor = torch.cat([feat_backdoor, weights_feat], dim=1)

            inputs = torch.cat([feat_benign, feat_backdoor], dim=0)  # [2, F]
            logits = model(inputs)
            predictions = (logits > 0).squeeze().long()
            labels = torch.tensor([0, 1], device=device)  # 0 for benign, 1 for backdoor
            if predictions[0] == labels[0]:
                tn_count += 1
            else:
                fp_count += 1

            if predictions[1] == labels[1]:
                tp_count += 1
            else:
                fn_count += 1
            if i % 10 == 0:
                print("tn_count:", tn_count, "tp_count:", tp_count, "fp_count:", fp_count, "fn_count:", fn_count)
    # 计算总样本数和概率
    total_benign = tn_count + fp_count
    total_backdoor = tp_count + fn_count
    total_samples = total_benign + total_backdoor

    if total_samples > 0:
        true_positive_rate = tp_count / total_backdoor if total_backdoor > 0 else 0
        true_negative_rate = tn_count / total_benign if total_benign > 0 else 0
        false_positive_rate = fp_count / total_benign if total_benign > 0 else 0
        false_negative_rate = fn_count / total_backdoor if total_backdoor > 0 else 0
        accuracy = (tp_count + tn_count) / total_samples
    else:
        true_positive_rate = true_negative_rate = false_positive_rate = false_negative_rate = accuracy = 0

    # 打印并保存结果
    results = {
        'True Positives': tp_count,
        'True Negatives': tn_count,
        'False Positives': fp_count,
        'False Negatives': fn_count,
        'True Positive Rate (TPR)': true_positive_rate,
        'True Negative Rate (TNR)': true_negative_rate,
        'False Positive Rate (FPR)': false_positive_rate,
        'False Negative Rate (FNR)': false_negative_rate,
        'Accuracy': accuracy
    }

    print("\n--- Evaluation Results ---")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    with open("evaluation_results_single_loop.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print("\nResults saved to evaluation_results_single_loop.txt")