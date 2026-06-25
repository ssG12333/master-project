import pandas as pd
import numpy as np
import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.cluster import kmeans_plusplus
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad
from tqdm import tqdm
import warnings
import time
import random
warnings.filterwarnings("ignore")
def set_seed(seed):
    """Sets seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
SEED = 888
set_seed(SEED)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
SOURCE_DATA_FILE = 'final_source_data.parquet'
TARGET_DATA_FILE = 'final_target_data.parquet'
SELECTED_FEATURES_FILE = 'selected_features.json'
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = f'output_run_{TIMESTAMP}'
CLASS_MAPPING = {'N': 0, 'B': 1, 'IR': 2, 'OR': 3}
REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
CLASS_LABELS = ['正常(N)', '滚动体故障(B)', '内圈故障(IR)', '外圈故障(OR)']
NUM_CLASSES = len(CLASS_LABELS)
PRETRAIN_EPOCHS = 60
ADDA_EPOCHS = 150
BATCH_SIZE = 256
LR_D = 1e-4
LR_G = 1e-4
CRITIC_ITERS = 5
LAMBDA_GP = 10
LAMBDA_CORAL = 0.5
LAMBDA_ENTROPY_MAX = 0.1
ENTROPY_WARMUP_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 30
NUM_BACKGROUND_SAMPLES = 200
MAX_FEATURES_TO_PLOT = 15
class FeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=128):
        super(FeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size * 2), nn.BatchNorm1d(hidden_size * 2), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU(), nn.Dropout(0.5)
        )
    def forward(self, x): return self.network(x)
class Classifier(nn.Module):
    def __init__(self, input_size=128, num_classes=4):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.network(x)
class Discriminator(nn.Module):
    def __init__(self, input_size=128, hidden_size=256):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size // 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, x): return self.network(x)
class SmoothedEarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', mode='min', smoothing_window=5):
        self.patience, self.verbose, self.delta, self.path, self.mode = patience, verbose, delta, path, mode
        self.counter, self.best_score, self.early_stop = 0, None, False
        self.val_loss_min = np.inf
        self.history = []
        self.smoothing_window = smoothing_window
    def __call__(self, val_loss, model):
        self.history.append(val_loss)
        if len(self.history) > self.smoothing_window: self.history.pop(0)
        smoothed_loss = np.mean(self.history)
        score = -smoothed_loss if self.mode == 'min' else smoothed_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(smoothed_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose and self.counter > 0 and self.counter % 10 == 0:
                 print(f'EarlyStopping counter: {self.counter}/{self.patience} (Smoothed Loss: {smoothed_loss:.6f})')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(smoothed_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
def coral_loss(source, target):
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)
    if ns < 2 or nt < 2: return torch.tensor(0.0, device=DEVICE)
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm) / (ns - 1)
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt) / (nt - 1)
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    return loss / (4 * d * d)
def compute_gradient_penalty(D, real_samples, fake_samples):
    batch_size = min(real_samples.size(0), fake_samples.size(0))
    real_samples, fake_samples = real_samples[:batch_size], fake_samples[:batch_size]
    alpha = torch.rand(batch_size, 1, device=DEVICE)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(batch_size, 1, requires_grad=False, device=DEVICE)
    gradients = grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()
def entropy_loss(p):
    p = torch.softmax(p, dim=1)
    return -torch.mean(torch.sum(p * torch.log(p + 1e-8), dim=1))
def plot_tsne_domain_comparison(X_source, X_target, title, full_path):
    print(f"正在生成 t-SNE 领域分布图: {title}...")
    n_source, n_target = X_source.shape[0], X_target.shape[0]
    X_combined = np.vstack((X_source, X_target))
    perplexity_val = min(30, X_combined.shape[0] - 1)
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity_val, max_iter=1000, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_combined)
    plt.figure(figsize=(16, 12));
    palette = {'源域': 'royalblue', '目标域': 'coral'}
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=['源域'] * n_source + ['目标域'] * n_target, palette=palette, style=['源域'] * n_source + ['目标域'] * n_target, s=50, alpha=0.7)
    plt.title(title, fontsize=20);
    plt.xlabel('t-SNE Dimension 1');
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='数据域');
    plt.grid(True);
    plt.tight_layout()
    plt.savefig(full_path);
    plt.close()
    print(f"t-SNE 图已保存至: {full_path}")
def plot_final_analysis(result_df, full_path):
    print("正在生成最终结果分析图...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [1, 2]})
    sns.barplot(x='置信度', y='文件名', data=result_df.sort_values('置信度', ascending=False), ax=axes[0], palette='summer')
    axes[0].set_title('各文件最终标签预测置信度', fontsize=16);
    axes[0].set_xlabel('平均最大概率');
    axes[0].set_xlim(0, 1.0)
    vote_counts = result_df.set_index('文件名')[[f"{REVERSE_CLASS_MAPPING[i]}_votes" for i in range(NUM_CLASSES)]]
    vote_counts.plot(kind='barh', stacked=True, ax=axes[1], colormap='viridis', width=0.8)
    axes[1].set_title('各文件内部样本段投票分布', fontsize=16);
    axes[1].set_xlabel('样本段数量');
    axes[1].set_ylabel('文件名')
    axes[1].legend(title='故障类型')
    plt.tight_layout();
    plt.savefig(full_path);
    plt.close()
    print(f"最终结果分析图已保存至: {full_path}")
def pretrain_source_classifier(input_dim, train_loader, val_loader, output_dir):
    print("\n" + "="*25 + " 阶段一: 源域预训练 " + "="*25)
    Fs, Cs = FeatureExtractor(input_dim).to(DEVICE), Classifier().to(DEVICE)
    optimizer = optim.AdamW(list(Fs.parameters()) + list(Cs.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = 0
    fs_path = os.path.join(output_dir, 'Fs_best.pt')
    cs_path = os.path.join(output_dir, 'Cs_best.pt')
    for epoch in range(PRETRAIN_EPOCHS):
        Fs.train(); Cs.train()
        for data, labels in train_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(); loss = loss_fn(Cs(Fs(data)), labels); loss.backward(); optimizer.step()

        Fs.eval(); Cs.eval(); correct, total = 0, 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                _, pred = torch.max(Cs(Fs(data)).data, 1)
                total += labels.size(0); correct += (pred == labels).sum().item()
        val_accuracy = 100 * correct / total
        print(f"预训练 Epoch [{epoch+1}/{PRETRAIN_EPOCHS}], 验证准确率: {val_accuracy:.2f}%")
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(Fs.state_dict(), fs_path); torch.save(Cs.state_dict(), cs_path)
            if val_accuracy > 99.5: print("准确率达标，提前结束。"); break
    Fs.load_state_dict(torch.load(fs_path)); Cs.load_state_dict(torch.load(cs_path))
    return Fs, Cs
def run_domain_adaptation(Fs, Cs, input_dim, source_loader, target_loader, output_dir):
    print("\n" + "="*25 + " 阶段二: 混合模型对抗适应 " + "="*25)
    Fs.eval(); Cs.eval()
    Ft = FeatureExtractor(input_dim).to(DEVICE); Ft.load_state_dict(Fs.state_dict())
    D = Discriminator().to(DEVICE)
    opt_D = optim.RMSprop(D.parameters(), lr=LR_D)
    opt_Ft = optim.Adam(Ft.parameters(), lr=LR_G, betas=(0.5, 0.9))
    history = {'d_loss': [], 'g_loss_adv': [], 'g_loss_coral': [], 'g_loss_entropy': []}
    ft_path = os.path.join(output_dir, 'Ft_best_adda.pt')
    early_stopper = SmoothedEarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True, path=ft_path, mode='min')
    epoch_pbar = tqdm(range(ADDA_EPOCHS), desc="混合模型训练")
    for epoch in epoch_pbar:
        D.train(); Ft.train()
        num_batches = min(len(source_loader), len(target_loader))
        source_iter, target_iter = iter(source_loader), iter(target_loader)
        lambda_entropy = LAMBDA_ENTROPY_MAX * min(1.0, epoch / ENTROPY_WARMUP_EPOCHS)
        d_loss_epoch_sum = 0
        for _ in range(num_batches):
            for _ in range(CRITIC_ITERS):
                s_data, _ = next(iter(source_loader)); t_data, _ = next(iter(target_loader))
                s_data, t_data = s_data.to(DEVICE), t_data.to(DEVICE)
                D.zero_grad()
                feat_s = Fs(s_data).detach(); feat_t = Ft(t_data).detach()
                d_loss = -torch.mean(D(feat_s)) + torch.mean(D(feat_t))
                gp = compute_gradient_penalty(D, feat_s.data, feat_t.data)
                d_loss_total = d_loss + LAMBDA_GP * gp
                d_loss_total.backward(); opt_D.step()
                d_loss_epoch_sum += d_loss.item()
            s_data_g, _ = next(iter(source_loader)); t_data_g, _ = next(iter(target_loader))
            s_data_g, t_data_g = s_data_g.to(DEVICE), t_data_g.to(DEVICE)
            common_bs = min(s_data_g.size(0), t_data_g.size(0))
            s_data_g, t_data_g = s_data_g[:common_bs], t_data_g[:common_bs]
            Ft.zero_grad()
            feat_t_fool = Ft(t_data_g); feat_s_new = Fs(s_data_g).detach()
            g_loss_adv = -torch.mean(D(feat_t_fool))
            g_loss_coral = coral_loss(feat_s_new, feat_t_fool)
            g_loss_entropy = entropy_loss(Cs(feat_t_fool))
            g_loss_total = g_loss_adv + LAMBDA_CORAL * g_loss_coral + lambda_entropy * g_loss_entropy
            g_loss_total.backward(); opt_Ft.step()
        avg_d_loss = d_loss_epoch_sum / (num_batches * CRITIC_ITERS)
        history['d_loss'].append(avg_d_loss)
        history['g_loss_adv'].append(g_loss_adv.item())
        history['g_loss_coral'].append(g_loss_coral.item())
        history['g_loss_entropy'].append(g_loss_entropy.item())
        pbar_postfix = {"W-Dist": f"{-avg_d_loss:.4f}", "G_adv": f"{g_loss_adv.item():.4f}", "G_coral": f"{g_loss_coral.item():.4f}"}
        epoch_pbar.set_postfix(pbar_postfix)
        early_stopper(-avg_d_loss, Ft)
        if early_stopper.early_stop: print("早停触发！"); break
    print(f"加载由早停机制保存的最佳 Ft 模型 (from {ft_path})...")
    Ft.load_state_dict(torch.load(ft_path))
    return Ft, history
def run_diagnosis_and_visualization(Fs, Ft, Cs, X_source_scaled, X_target_scaled, target_df, output_dir):
    print("\n" + "="*25 + " 阶段三: 最终诊断与可视化 " + "="*25)
    Ft.eval(); Cs.eval(); Fs.eval()
    with torch.no_grad():
        source_features = Fs(torch.FloatTensor(X_source_scaled).to(DEVICE)).cpu().numpy()
        target_features_aligned = Ft(torch.FloatTensor(X_target_scaled).to(DEVICE))
        post_transfer_outputs = Cs(target_features_aligned)
        post_transfer_probs = torch.softmax(post_transfer_outputs, dim=1).cpu().numpy()
        post_transfer_preds = np.argmax(post_transfer_probs, axis=1)
    plot_tsne_domain_comparison(source_features, target_features_aligned.cpu().numpy(),
                               't-SNE 领域分布 [迁移后]',
                               os.path.join(output_dir, 'q3_tsne_domain_after.png'))

    prediction_df = pd.DataFrame({'filename': target_df['filename'], 'prediction': post_transfer_preds, 'probability': np.max(post_transfer_probs, axis=1)})
    file_results = []
    for filename in sorted(target_df['filename'].unique()):
        base_name = os.path.splitext(filename)[0]
        file_df = prediction_df[prediction_df['filename'] == filename]
        vote_counts = Counter(file_df['prediction'])
        final_pred_int = vote_counts.most_common(1)[0][0]
        confidence = file_df[prediction_df['prediction'] == final_pred_int]['probability'].mean()
        votes_dict = {f"{REVERSE_CLASS_MAPPING[i]}_votes": vote_counts.get(i, 0) for i in range(NUM_CLASSES)}
        file_results.append({'文件名': base_name, '预测故障类型': REVERSE_CLASS_MAPPING[final_pred_int], '置信度': confidence, **votes_dict})
    result_df = pd.DataFrame(file_results)
    print("\n" + "="*25 + " 最终目标域文件标签标定结果 " + "="*25)
    print(result_df[['文件名', '预测故障类型', '置信度']].to_string(index=False))
    print("="*60)
    plot_final_analysis(result_df, os.path.join(output_dir, 'q3_final_analysis.png'))
    return post_transfer_preds
def run_interpretability_analysis(model, X_source_train_tensor, X_target_tensor, target_pred_labels, selected_features, output_dir):
    print("\n" + "="*25 + " 阶段四: 可解释性分析 (SHAP) " + "="*25)
    sample_indices = []
    unique_preds = np.unique(target_pred_labels)
    for i in range(len(CLASS_LABELS)):
        if i in unique_preds:
            idx = np.where(target_pred_labels == i)[0][0]; sample_indices.append(idx)
            print(f"为类别 '{CLASS_LABELS[i]}' 选择样本索引: {idx}")
    samples_to_explain = X_target_tensor[sample_indices]
    print(f"正在使用 K-Means++ 选择 {NUM_BACKGROUND_SAMPLES} 个代表性样本作为背景数据...")
    _, indices = kmeans_plusplus(X_source_train_tensor.numpy(), n_clusters=NUM_BACKGROUND_SAMPLES, random_state=SEED)
    background_data = X_source_train_tensor[indices].to(DEVICE)
    explainer = shap.GradientExplainer(model, background_data)
    with torch.no_grad(): expected_values = model(background_data).mean(0).cpu().numpy()
    print(f"正在为 {len(samples_to_explain)} 个样本计算SHAP值...")
    shap_values = explainer.shap_values(samples_to_explain.to(DEVICE))
    print("--- 正在可视化并保存SHAP瀑布图 (局部解释) ---")
    sorted_unique_preds = sorted(unique_preds)
    class_id_to_shap_idx = {class_id: i for i, class_id in enumerate(sorted_unique_preds)}
    for i, sample_idx in enumerate(sample_indices):
        pred_class_idx = target_pred_labels[sample_idx]
        pred_class_name = CLASS_LABELS[pred_class_idx]
        if pred_class_idx not in class_id_to_shap_idx: continue
        shap_idx = class_id_to_shap_idx[pred_class_idx]
        explanation = shap.Explanation(
            values=shap_values[shap_idx][i], base_values=expected_values[pred_class_idx],
            data=samples_to_explain[i].numpy(), feature_names=selected_features
        )
        plt.figure()
        shap.plots.waterfall(explanation, max_display=MAX_FEATURES_TO_PLOT, show=False)
        plt.title(f'样本 {sample_idx} 预测为 "{pred_class_name}" 的SHAP瀑布图')
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'q4_shap_waterfall_sample_{sample_idx}.png')
        plt.savefig(save_path); plt.close()
        print(f"瀑布图已保存至: {save_path}")
    print("\n--- 正在可视化并保存SHAP摘要图 (全局解释) ---")
    filtered_class_labels = [CLASS_LABELS[i] for i in sorted_unique_preds]
    samples_to_explain_np = samples_to_explain.numpy()
    plt.figure()
    shap.summary_plot(shap_values, samples_to_explain_np, feature_names=selected_features,
                      class_names=filtered_class_labels, max_display=MAX_FEATURES_TO_PLOT, show=False)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'q4_shap_summary_beeswarm_overall.png')
    plt.savefig(save_path); plt.close()
    print(f"全局蜂群图已保存至: {save_path}")
    for i, class_idx in enumerate(sorted_unique_preds):
        class_name = CLASS_LABELS[class_idx]
        plt.figure()
        shap.summary_plot(shap_values[i], feature_names=selected_features, plot_type="bar",
                          max_display=MAX_FEATURES_TO_PLOT, show=False)
        plt.title(f'对预测 "{class_name}" 类别最重要的特征')
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'q4_shap_summary_bar_class_{class_idx}.png')
        plt.savefig(save_path); plt.close()
        print(f"类别 '{class_name}' 的摘要条形图已保存至: {save_path}")
def main():
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")
    print("\n" + "="*25 + " 步骤零: 数据加载与准备 " + "="*25)
    source_df = pd.read_parquet(SOURCE_DATA_FILE)
    target_df = pd.read_parquet(TARGET_DATA_FILE)
    with open(SELECTED_FEATURES_FILE, 'r') as f:
        selected_features = json.load(f)
    X_source, y_source = source_df[selected_features].values, source_df['fault_type'].map(CLASS_MAPPING).values
    X_target = target_df[selected_features].values
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    input_dim = X_source_scaled.shape[1]
    X_source_train, X_source_val, y_source_train, y_source_val = train_test_split(
        X_source_scaled, y_source, test_size=0.2, random_state=SEED, stratify=y_source)
    source_train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_source_train), torch.LongTensor(y_source_train)), batch_size=BATCH_SIZE, shuffle=True)
    source_val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_source_val), torch.LongTensor(y_source_val)), batch_size=BATCH_SIZE)
    source_loader_adda = DataLoader(TensorDataset(torch.FloatTensor(X_source_scaled), torch.LongTensor(y_source)), batch_size=BATCH_SIZE, shuffle=True)
    target_loader_adda = DataLoader(TensorDataset(torch.FloatTensor(X_target_scaled), torch.zeros(len(X_target_scaled), dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=True)
    X_source_tensor = torch.FloatTensor(X_source_scaled)
    X_target_tensor = torch.FloatTensor(X_target_scaled)
    Fs, Cs = pretrain_source_classifier(input_dim, source_train_loader, source_val_loader, OUTPUT_DIR)
    Ft, history = run_domain_adaptation(Fs, Cs, input_dim, source_loader_adda, target_loader_adda, OUTPUT_DIR)
    target_pred_labels = run_diagnosis_and_visualization(Fs, Ft, Cs, X_source_scaled, X_target_scaled, target_df, OUTPUT_DIR)
    final_model = nn.Sequential(Ft, Cs).to(DEVICE).eval()
    run_interpretability_analysis(final_model, X_source_tensor, X_target_tensor, target_pred_labels, selected_features, OUTPUT_DIR)
    print(f"\n所有任务已完成！结果保存在目录: {OUTPUT_DIR}")
if __name__ == '__main__':
    main()

