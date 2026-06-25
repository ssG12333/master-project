import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score
from preprocessing import TBDataset, get_transforms, load_segmentation_model
from model import TBTransformerModel
from train import Config, evaluate_model


def evaluate_and_plot(config):
    # 加载数据集
    train_transform, val_transform = get_transforms(config.image_size)
    # 加载分割模型（如果需要）
    segmentor = None
    if config.use_segmentation:
        segmentor = load_segmentation_model()
    # 创建测试数据集
    test_dataset = TBDataset(
        root_dir=config.data_dir,
        split='test',
        transform=val_transform,
        label_file=config.label_file,
        use_segmentation=config.use_segmentation,
        segmentor=segmentor,
    )
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    # 初始化模型
    model = TBTransformerModel(
        num_classes=config.num_classes,
        use_attention=config.use_attention,
        pretrained=config.use_pretrained,
        dropout_rate=config.dropout_rate
    ).to(config.device)
    # 加载最佳模型
    model.load_state_dict(torch.load(config.best_model_path))
    # 在测试集上评估模型
    test_loss, test_auc, test_labels, test_preds = evaluate_model(model, test_loader, config.device,
                                                                  torch.nn.CrossEntropyLoss(),
                                                                  phase="test")

    # 计算准确度
    test_preds_binary = np.round(test_preds)  # 将预测概率转换为二元标签（0或1）
    accuracy = accuracy_score(test_labels, test_preds_binary)

    print(f"测试损失: {test_loss:.4f} | 测试AUC: {test_auc:.4f} | 测试准确度: {accuracy:.4f}")

    # 绘制 ROC 曲线
    fpr, tpr, _ = roc_curve(test_labels, test_preds)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(config.save_dir, 'roc_curve.png'))
    plt.close()

    # 绘制 PR 曲线
    precision, recall, _ = precision_recall_curve(test_labels, test_preds)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='PR curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(config.save_dir, 'pr_curve.png'))
    plt.close()


if __name__ == "__main__":
    config = Config()
    evaluate_and_plot(config)