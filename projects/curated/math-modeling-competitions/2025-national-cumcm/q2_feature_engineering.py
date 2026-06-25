import pandas as pd
import numpy as np
import json
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             f1_score, recall_score)
from sklearn.base import BaseEstimator, ClassifierMixin

import xgboost as xgb # 重新导入xgboost以用作元模型
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

SOURCE_DATA_FILE = 'final_source_data.parquet'
SELECTED_FEATURES_FILE = 'selected_features.json'
OUTPUT_DIR = 'plots_q2最最终'
CLASS_MAPPING = {'N': 0, 'B': 1, 'IR': 2, 'OR': 3}
CLASS_LABELS = ['正常(N)', '滚动体故障(B)', '内圈故障(IR)', '外圈故障(OR)']
NUM_CLASSES = len(CLASS_LABELS)
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.network(x)
class FeatureAttention(nn.Module):
    def __init__(self, input_dim):
        super(FeatureAttention, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        attention_weights = self.attention_net(x)
        return x * attention_weights
class MLPWithAttention(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPWithAttention, self).__init__()
        self.attention = FeatureAttention(input_size)
        self.classifier = MLP(input_size, num_classes)
    def forward(self, x):
        weighted_features = self.attention(x)
        return self.classifier(weighted_features)
class SklearnNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class, input_size, num_classes, epochs=50, lr=0.001, batch_size=64):
        self.model_class = model_class
        self.input_size = input_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = self.model_class(self.input_size, self.num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()
    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()
def plot_confusion_matrix(y_true, y_pred, labels, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'混淆矩阵 - {model_name} (在独立测试集上)', fontsize=16)
    plt.ylabel('真实标签');
    plt.xlabel('预测标签')
    save_path = os.path.join(OUTPUT_DIR, f"q2_cm_{model_name.replace(' ', '_')}.png")
    plt.savefig(save_path);
    plt.close()
    print(f"混淆矩阵图已保存至: {save_path}")
def plot_metric_comparison(results_df, metric_name, title):
    plt.figure(figsize=(12, 8))
    sorted_df = results_df.sort_values(metric_name, ascending=False)
    sns.barplot(x=metric_name, y='模型名称', data=sorted_df, palette='viridis')
    plt.title(title, fontsize=16)
    plt.xlabel(metric_name);
    plt.ylabel('模型')
    plt.xlim(min(sorted_df[metric_name]) * 0.9, 1.0)
    for index, value in enumerate(sorted_df[metric_name]):
        plt.text(value, index, f'{value:.4f}', va='center')
    plt.tight_layout()
    filename = f"q2_comparison_{metric_name.lower().replace(' (test)', '').replace(' ', '_')}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path);
    plt.close()
    print(f"\n模型 {metric_name} 对比图已保存至: {save_path}")
def train_and_evaluate_on_test_set(model, X_train, y_train, X_test, y_test, model_name):
    print(f"\n{'=' * 20} 在测试集上评估: {model_name} {'=' * 20}")
    start_time = time.time()
    model.fit(X_train, y_train)
    print(f"模型训练完成。耗时: {time.time() - start_time:.2f} 秒")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\n--- {model_name} 在独立测试集上的最终性能 ---")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"宏平均召回率 (Macro Recall): {recall:.4f}")
    print(f"宏平均 F1 分数 (Macro F1): {f1:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS, digits=4))
    plot_confusion_matrix(y_test, y_pred, CLASS_LABELS, model_name)
    return accuracy, recall, f1, model
if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print("--- 步骤 1: 加载数据 ---")
    source_df = pd.read_parquet(SOURCE_DATA_FILE)
    with open(SELECTED_FEATURES_FILE, 'r') as f:
        selected_features = json.load(f)
    X = source_df[selected_features].fillna(0).values
    y = source_df['fault_type'].map(CLASS_MAPPING).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("数据加载与预处理完成。")
    input_dim = X_train_scaled.shape[1]
    models = {
        "逻辑回归": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "随机森林": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "支持向量机": SVC(random_state=42, probability=True),
        "LightGBM": lgb.LGBMClassifier(objective='multiclass', num_class=NUM_CLASSES, n_jobs=-1, random_state=40),
        "自定义MLP": SklearnNNWrapper(MLP, input_size=input_dim, num_classes=NUM_CLASSES),
        "自定义注意力MLP": SklearnNNWrapper(MLPWithAttention, input_size=input_dim, num_classes=NUM_CLASSES)
    }
    print("\n--- 步骤 2: 在独立测试集上进行模型评估 ---")
    final_results = []
    trained_models = {}
    for name, model in models.items():
        accuracy, recall, f1, trained_model = train_and_evaluate_on_test_set(
            model, X_train_scaled, y_train, X_test_scaled, y_test, name
        )
        final_results.append({
            "模型名称": name, "准确率 (Test)": accuracy,
            "召回率 (Test)": recall, "F1分数 (Test)": f1
        })
        trained_models[name] = trained_model
    temp_df = pd.DataFrame(final_results)
    top_3_model_names = temp_df.sort_values('F1分数 (Test)', ascending=False).head(3)['模型名称'].tolist()
    print(f"\n选择F1分数排名前三的模型进行Stacking: {top_3_model_names}")
    estimators_for_stacking = [(name, models[name]) for name in top_3_model_names]
    meta_model = xgb.XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', n_jobs=-1, random_state=42, use_label_encoder=False)
    stacking_model = StackingClassifier(estimators=estimators_for_stacking, final_estimator=meta_model, cv=5, n_jobs=-1)
    stacking_accuracy, stacking_recall, stacking_f1, _ = train_and_evaluate_on_test_set(
        stacking_model, X_train_scaled, y_train, X_test_scaled, y_test, "Stacking (Top 3)"
    )
    final_results.append({
        "模型名称": "Stacking (Top 3)", "准确率 (Test)": stacking_accuracy,
        "召回率 (Test)": stacking_recall, "F1分数 (Test)": stacking_f1
    })
    print(f"\n{'=' * 25} 所有模型在独立测试集上的最终结果 {'=' * 25}")
    final_results_df = pd.DataFrame(final_results)
    print(final_results_df.to_string(index=False))
    plot_metric_comparison(final_results_df, '准确率 (Test)', '不同模型在独立测试集上的【准确率】对比')
    plot_metric_comparison(final_results_df, '召回率 (Test)', '不同模型在独立测试集上的【宏平均召回率】对比')
    plot_metric_comparison(final_results_df, 'F1分数 (Test)', '不同模型在独立测试集上的【宏平均F1分数】对比')
    print("\n第二问所有任务已完成！")