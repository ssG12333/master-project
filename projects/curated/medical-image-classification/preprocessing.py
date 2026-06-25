import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn.functional as F
import random
from collections import Counter
import matplotlib
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题

# 设置随机种子确保结果可复现
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MedicalTransform:
    """医学图像专用预处理和增强变换"""

    def __init__(self, image_size=224, is_train=True, use_albumentations=True):
        self.image_size = image_size
        self.is_train = is_train
        self.use_albumentations = use_albumentations

        # 通用归一化参数
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if use_albumentations:
            self.train_transform = A.Compose([
                A.Resize(image_size, image_size),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.GaussianBlur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2, p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                ], p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), p=0.5),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])

            self.val_transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
        else:import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn.functional as nnf
import random
from collections import Counter
import matplotlib
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class TBDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, label_file=None,
                 use_segmentation=False, segmentor=None, remove_duplicates=True,
                 remove_ambiguous=True):
        """
        肺结核数据集处理类
        参数:
            root_dir: 数据集根目录（TB_dataset 路径）
            split: 数据集划分，可选'train', 'val', 'test'
            transform: 图像预处理和增强变换
            label_file: 标签文件路径
            use_segmentation: 是否使用肺部分割
            segmentor: 肺部分割模型
            remove_duplicates: 是否移除重复样本
            remove_ambiguous: 是否移除矛盾标签样本
        """
        self.transform = transform
        self.split = split
        self.use_segmentation = use_segmentation
        self.segmentor = segmentor

        # 数据集路径
        self.root_dir = os.path.join(root_dir, "dataset", split)

        # 检查路径是否存在
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"路径不存在: {self.root_dir}")

        # 加载图像文件列表
        self.image_files = []
        for file in os.listdir(self.root_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_files.append(os.path.join(self.root_dir, file))

        print(f"{split}集原始图像数量: {len(self.image_files)}")

        # 初始化标签相关属性
        self.labels_df = None
        self.img_to_label = {}

        if label_file:
            # 加载标签文件
            self.labels_df = pd.read_csv(label_file)

            # 移除重复样本
            if remove_duplicates and 'patient_id' in self.labels_df.columns:
                original_count = len(self.labels_df)
                self.labels_df = self.labels_df.drop_duplicates(subset='patient_id')
                print(f"移除重复样本: {original_count} -> {len(self.labels_df)}")

            # 移除矛盾标签（同时包含"No Finding"和其他标签）
            if remove_ambiguous and 'Finding Labels' in self.labels_df.columns:
                original_count = len(self.labels_df)
                ambiguous_mask = self.labels_df['Finding Labels'].str.contains('No Finding') & \
                                 self.labels_df['Finding Labels'].str.contains(r'\|')
                self.labels_df = self.labels_df[~ambiguous_mask]
                print(f"移除矛盾标签: {original_count} -> {len(self.labels_df)}")

            # 定义肺结核相关的阳性标签（全部小写，避免大小写问题）
            tb_positive_labels = ['infiltration', 'consolidation', 'pneumonia', 'effusion', 'fibrosis']

            # 映射为二分类标签：包含任一阳性标签则为1，否则为0
            # 使用小写匹配，提高鲁棒性
            self.labels_df['tb_label'] = self.labels_df['Finding Labels'].apply(
                lambda x: 1 if any(label in x.lower().split('|') for label in tb_positive_labels) else 0
            )

            # 构建图像名称到标签的映射
            if 'Image Index' in self.labels_df.columns:
                # 确保标签DataFrame中的Image Index列没有重复值
                if self.labels_df['Image Index'].duplicated().any():
                    print("警告: 标签文件中存在重复的Image Index，将保留第一个出现的值")
                    self.labels_df = self.labels_df.drop_duplicates(subset='Image Index')

                self.img_to_label = {row['Image Index']: row['tb_label']
                                     for _, row in self.labels_df.iterrows()}
                print(f"构建的标签映射数量: {len(self.img_to_label)}")

                # 分析标签分布
                label_distribution = self.labels_df['tb_label'].value_counts()
                print(f"完整标签文件中的标签分布:")
                print(label_distribution)

                # 分析标签文件中各类别的图像是否存在于当前split中
                available_images = set(os.path.basename(f) for f in self.image_files)
                positive_images_in_split = [img for img, label in self.img_to_label.items()
                                            if label == 1 and img in available_images]
                negative_images_in_split = [img for img, label in self.img_to_label.items()
                                            if label == 0 and img in available_images]

                print(f"当前{split}集中可用的阳性样本: {len(positive_images_in_split)}")
                print(f"当前{split}集中可用的阴性样本: {len(negative_images_in_split)}")

                # 如果没有找到阳性样本，打印详细信息帮助诊断
                if len(positive_images_in_split) == 0:
                    print("警告: 当前数据集中没有找到阳性样本！")
                    print("可能的原因:")
                    print("1. 标签映射逻辑不正确")
                    print("2. 阳性样本没有被分配到当前split中")
                    print("3. 图像文件名与标签文件中的Image Index不匹配")

                    # 打印一些阳性标签的样本，帮助诊断
                    positive_samples = self.labels_df[self.labels_df['tb_label'] == 1].sample(
                        min(5, len(self.labels_df[self.labels_df['tb_label'] == 1])))
                    print("\n阳性标签样本示例:")
                    print(positive_samples[['Image Index', 'Finding Labels', 'tb_label']])

                # 过滤不在标签文件中的图像
                original_count = len(self.image_files)
                self.image_files = [f for f in self.image_files
                                    if os.path.basename(f) in self.img_to_label]
                filtered_count = len(self.image_files)
                print(f"过滤后保留的图像数量: {filtered_count} ({original_count - filtered_count} 张图像被移除)")

                # 检查是否有标签但没有对应的图像
                image_names = set(os.path.basename(f) for f in self.image_files)
                labels_without_images = [name for name in self.img_to_label if name not in image_names]
                print(f"有标签但无对应图像的数量: {len(labels_without_images)}")

                # 检查图像标签分布
                if self.image_files:
                    labels = [self.img_to_label[os.path.basename(f)] for f in self.image_files]
                    label_distribution = Counter(labels)
                    print(f"{split}集过滤后的标签分布: {label_distribution}")
            else:
                print("警告: 标签文件中没有'Image Index'列，无法建立图像与标签的映射")
        else:
            # 如果没有提供标签文件，使用文件名作为标签（假设文件名包含标签信息）
            self.img_to_label = {os.path.basename(f): 1 if 'tb' in f.lower() else 0
                                 for f in self.image_files}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 添加索引检查
        if idx >= len(self.image_files):
            raise IndexError(f"索引 {idx} 超出范围，数据集大小为 {len(self.image_files)}")

        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        # 应用肺部分割（如果启用）
        lung_mask = torch.zeros(1)  # 初始化肺掩码
        if self.use_segmentation and self.segmentor:
            original_size = image.size
            # 预处理图像用于分割模型
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0)
            self.segmentor.eval()
            with torch.no_grad():
                output = self.segmentor(input_tensor)['out'][0]
            mask = output.argmax(0).byte().cpu().numpy()
            mask_pil = Image.fromarray(mask * 255).resize(original_size, Image.NEAREST)
            image = Image.composite(
                image,
                Image.new('RGB', original_size, (0, 0, 0)),
                mask_pil
            )
            lung_mask = torch.tensor(np.array(mask_pil), dtype=torch.float32).unsqueeze(0)  # 转换为张量

        # 应用预处理和增强变换
        if self.transform:
            image = self.transform(image)

        # 获取标签
        label = self.img_to_label.get(os.path.basename(img_path), 0)
        return image, label, img_path, lung_mask

    def _segment_lungs(self, image):
        """使用预训练的U-Net模型进行肺部分割"""
        # 保存原始尺寸用于后续恢复
        original_size = image.size

        # 预处理图像用于分割模型
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 转换为batch格式
        input_tensor = transform(image).unsqueeze(0)

        # 使用分割模型预测
        self.segmentor.eval()
        with torch.no_grad():
            output = self.segmentor(input_tensor)['out'][0]

        # 获取预测的掩码（假设类别1是肺部区域）
        mask = output.argmax(0).byte().cpu().numpy()

        # 转换回PIL图像并调整到原始尺寸
        mask_pil = Image.fromarray(mask * 255).resize(original_size, Image.NEAREST)

        # 应用掩码到原始图像
        masked_image = Image.composite(
            image,
            Image.new('RGB', original_size, (0, 0, 0)),
            mask_pil
        )
        return masked_image

    def check_label_mapping(self):
        pass


def load_segmentation_model(model_path=None):
    """加载预训练的肺部分割模型"""
    # 使用DeepLabv3作为分割模型
    model = deeplabv3_resnet50(weights=None, num_classes=2)

    # 如果有预训练权重，加载它们
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    return model


def get_transforms(image_size=224):
    """定义训练集和验证集的数据预处理和增强变换"""
    # 训练集变换（包含增强）
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证/测试集变换（仅基本预处理）
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def create_datasets(data_dir, label_file=None, use_segmentation=False, segmentor=None):
    """创建训练集、验证集和测试集"""
    train_transform = get_transforms(augment=True)
    val_test_transform = get_transforms(augment=False)

    train_dataset = TBDataset(
        root_dir=data_dir,
        split='train',
        transform=train_transform,
        label_file=label_file,
        use_segmentation=use_segmentation,
        segmentor=segmentor
    )

    val_dataset = TBDataset(
        root_dir=data_dir,
        split='val',
        transform=val_test_transform,
        label_file=label_file,
        use_segmentation=use_segmentation,
        segmentor=segmentor
    )

    test_dataset = TBDataset(
        root_dir=data_dir,
        split='test',
        transform=val_test_transform,
        label_file=label_file,
        use_segmentation=use_segmentation,
        segmentor=segmentor
    )

    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=4):
    """创建数据加载器"""
    # 训练集使用加权采样器处理类别不平衡
    if hasattr(train_dataset, 'labels_df') and len(train_dataset) > 0:
        # 获取训练集的实际标签分布
        labels = []
        print("正在计算训练集实际标签分布...")
        for i in tqdm(range(len(train_dataset))):
            _, label, _ = train_dataset[i]
            labels.append(label)

        label_counts = Counter(labels)
        print(f"训练集实际标签分布: {label_counts}")

        # 计算样本权重
        if len(label_counts) > 1:  # 确保有多个类别
            class_sample_count = np.array([label_counts[c] for c in sorted(label_counts.keys())])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[label] for label in labels])
            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()

            # 创建加权随机采样器
            sampler = WeightedRandomSampler(
                samples_weight, len(samples_weight)
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            print("警告: 训练集中只有一个类别，使用普通随机采样")
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
    else:
        # 如果没有标签数据或数据集为空，使用普通随机采样
        print("警告: 无法获取训练集标签分布，使用普通随机采样")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def visualize_samples(loader, title, num_samples=9):
    """可视化样本"""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    if len(loader.dataset) == 0:
        print(f"警告：{title} 数据加载器为空！")
        return

    # 尝试获取样本
    try:
        print(f"尝试从{title}加载器获取样本...")
        images, labels, paths = next(iter(loader))
        print(f"成功获取{len(images)}个样本")

        # 打印前几个样本的路径和标签，帮助调试
        print(f"{title}样本示例:")
        for i in range(min(5, len(paths))):
            print(f"  {os.path.basename(paths[i])}: 标签={labels[i].item()}")

    except Exception as e:
        print(f"获取样本失败: {e}")
        print(f"数据集大小: {len(loader.dataset)}")

        # 尝试逐个获取样本，找出问题
        print("尝试逐个获取样本以诊断问题:")
        success = 0
        failed = 0
        for i in range(min(10, len(loader.dataset))):
            try:
                _, _, path = loader.dataset[i]
                print(f"  样本 {i}: {os.path.basename(path)} - 成功")
                success += 1
            except Exception as ex:
                print(f"  样本 {i}: 失败 - {ex}")
                failed += 1

        print(f"成功: {success}, 失败: {failed}")
        return

    # 反归一化处理
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean

    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    with tqdm(total=num_samples, desc="可视化样本") as pbar:
        for i in range(num_samples):
            img = images[i].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[i].imshow(img)
            axes[i].set_title(f"Label: {labels[i].item()}")
            axes[i].axis('off')
            pbar.update(1)

    plt.suptitle(title)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(save_path)
    print(f"图像已保存至：{save_path}")
    plt.close()


def analyze_dataset(dataset, split_name):
    """分析数据集的类别分布"""
    if hasattr(dataset, 'labels_df'):
        # 统计数据集中的实际标签分布
        labels = []
        print(f"正在分析{split_name}实际标签分布...")
        for i in tqdm(range(len(dataset))):
            _, label, _ = dataset[i]
            labels.append(label)

        label_counts = Counter(labels)
        total = len(labels)

        print(f"{split_name} 数据集实际标签分布:")
        for label, count in label_counts.items():
            print(f"  类别 {label}: {count} ({count / total * 100:.2f}%)")

        return label_counts
    else:
        print(f"{split_name} 数据集: 无法分析类别分布（没有标签数据）")
        return None


def main():
    # 数据集根目录和标签文件路径（TB_dataset 路径）
    data_dir = r"D:\PycharmProjects\PythonProject1\TB_dataset"
    label_file = os.path.join(data_dir, "Data_Entry_2017.csv")

    # 加载标签文件并查看原始标签分布
    if os.path.exists(label_file):
        labels_df = pd.read_csv(label_file)
        print("原始标签分布:")
        print(labels_df['Finding Labels'].value_counts())

        # 定义肺结核相关的阳性标签（全部小写）
        tb_positive_labels = ['infiltration', 'consolidation', 'pneumonia', 'effusion', 'fibrosis']

        # 映射为二分类标签：包含任一阳性标签则为1，否则为0
        labels_df['tb_label'] = labels_df['Finding Labels'].apply(
            lambda x: 1 if any(label in x.lower().split('|') for label in tb_positive_labels) else 0
        )

        print("\n映射后的二分类标签分布:")
        print(labels_df['tb_label'].value_counts())

        # 验证标签映射示例
        print("\n验证标签映射示例:")
        sample_data = labels_df.sample(5)[['Finding Labels', 'tb_label']]
        print(sample_data)

        # 分析标签映射的准确性
        positive_samples = labels_df[labels_df['tb_label'] == 1]
        print(f"\n阳性样本示例 ({len(positive_samples)} 个):")
        print(positive_samples[['Finding Labels', 'tb_label']].head().to_string())

        # 创建数据集（不使用分割模型，先验证标签）
        use_segmentation = False
        segmentor = None
        print("\n正在创建数据集...")
        train_dataset, val_dataset, test_dataset = create_datasets(
            data_dir, label_file, use_segmentation, segmentor
        )

        # 分析数据集类别分布
        analyze_dataset(train_dataset, "训练集")
        analyze_dataset(val_dataset, "验证集")
        analyze_dataset(test_dataset, "测试集")

        # 创建数据加载器
        print("\n正在创建数据加载器...")
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset, batch_size=16
        )

        # 可视化样本
        if len(train_loader.dataset) > 0:
            print("\n开始可视化训练集样本...")
            visualize_samples(train_loader, "训练集样本")
            print("训练集可视化完成！")
        else:
            print("警告：训练集数据加载器为空！")

        if len(val_loader.dataset) > 0:
            print("\n开始可视化验证集样本...")
            visualize_samples(val_loader, "验证集样本")
            print("验证集可视化完成！")
        else:
            print("警告：验证集数据加载器为空！")

        if len(test_loader.dataset) > 0:
            print("\n开始可视化测试集样本...")
            visualize_samples(test_loader, "测试集样本")
            print("测试集可视化完成！")
        else:
            print("警告：测试集数据加载器为空！")

        print(f"\n数据集大小:")
        print(f"训练集: {len(train_dataset)}")
        print(f"验证集: {len(val_dataset)}")
        print(f"测试集: {len(test_dataset)}")
    else:
        print(f"标签文件不存在: {label_file}")


if __name__ == "__main__":
    main()


def get_transform():
    return None

        # 传统torchvision变换
    self.train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])

    self.val_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])

    def __call__(self, image):
        if self.use_albumentations:
            image = np.array(image)
            if self.is_train:
                transformed = self.train_transform(image=image)
            else:
                transformed = self.val_transform(image=image)
            return transformed['image']
        else:
            if self.is_train:
                return self.train_transform(image)
            else:
                return self.val_transform(image)


class TBDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, label_file=None,
                 use_segmentation=False, segmentor=None, remove_duplicates=True,
                 remove_ambiguous=True, clinical_data_file=None):
        """
        改进的肺结核数据集处理类
        新增功能:
        1. 支持临床数据融合
        2. 更健壮的标签处理
        3. 优化的图像预处理流程
        """
        self.split = split
        self.use_segmentation = use_segmentation
        self.segmentor = segmentor

        # 初始化变换
        self.transform = transform if transform else MedicalTransform(is_train=(split == 'train'))

        # 数据集路径
        self.root_dir = os.path.join(root_dir, "dataset", split)
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"路径不存在: {self.root_dir}")

        # 加载图像文件列表
        self.image_files = sorted([
            os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        print(f"{split}集原始图像数量: {len(self.image_files)}")

        # 初始化标签和临床数据
        self.labels_df = None
        self.clinical_df = None
        self.img_to_label = {}
        self.img_to_clinical = {}

        # 加载标签文件
        if label_file:
            self._load_label_data(label_file, remove_duplicates, remove_ambiguous)

        # 加载临床数据
        if clinical_data_file:
            self._load_clinical_data(clinical_data_file)

    def _load_label_data(self, label_file, remove_duplicates, remove_ambiguous):
        """加载并处理标签数据"""
        self.labels_df = pd.read_csv(label_file)

        # 数据清洗
        original_count = len(self.labels_df)

        # 移除重复样本
        if remove_duplicates and 'patient_id' in self.labels_df.columns:
            self.labels_df = self.labels_df.drop_duplicates(subset='patient_id')
            print(f"移除重复样本: {original_count} -> {len(self.labels_df)}")

        # 移除矛盾标签
        if remove_ambiguous and 'Finding Labels' in self.labels_df.columns:
            ambiguous_mask = self.labels_df['Finding Labels'].str.contains('No Finding') & \
                             self.labels_df['Finding Labels'].str.contains(r'\|')
            self.labels_df = self.labels_df[~ambiguous_mask]
            print(f"移除矛盾标签: {original_count} -> {len(self.labels_df)}")

        # 肺结核阳性标签定义
        tb_positive_labels = ['infiltration', 'consolidation', 'pneumonia', 'effusion', 'fibrosis']

        # 改进的标签映射逻辑
        def map_label(x):
            labels = x.lower().split('|')
            # 优先检查明确标签
            if 'tuberculosis' in labels:
                return 1
            if 'no finding' in labels:
                return 0
            # 检查其他阳性指标
            return 1 if any(label in labels for label in tb_positive_labels) else 0

        self.labels_df['tb_label'] = self.labels_df['Finding Labels'].apply(map_label)

        # 构建图像名称到标签的映射
        if 'Image Index' in self.labels_df.columns:
            if self.labels_df['Image Index'].duplicated().any():
                print("警告: 标签文件中存在重复的Image Index，将保留第一个出现的值")
                self.labels_df = self.labels_df.drop_duplicates(subset='Image Index')

            self.img_to_label = {
                row['Image Index']: row['tb_label']
                for _, row in self.labels_df.iterrows()
            }
            print(f"构建的标签映射数量: {len(self.img_to_label)}")

            # 分析标签分布
            label_distribution = self.labels_df['tb_label'].value_counts()
            print(f"完整标签文件中的标签分布:\n{label_distribution}")

            # 过滤不在标签文件中的图像
            self._filter_images_by_labels()

            # 检查标签映射
            self._check_label_mapping()

    def _load_clinical_data(self, clinical_data_file):
        """加载临床数据"""
        self.clinical_df = pd.read_csv(clinical_data_file)

        # 简单处理: 假设有'Image Index'列和临床特征列
        if 'Image Index' in self.clinical_df.columns:
            # 选择数值型临床特征
            clinical_features = self.clinical_df.select_dtypes(include=['number']).columns.tolist()
            clinical_features = [f for f in clinical_features if f != 'Image Index']

            # 标准化临床数据
            for feature in clinical_features:
                mean = self.clinical_df[feature].mean()
                std = self.clinical_df[feature].std()
                self.clinical_df[feature] = (self.clinical_df[feature] - mean) / std

            # 构建图像到临床数据的映射
            self.img_to_clinical = {
                row['Image Index']: row[clinical_features].values.astype(np.float32)
                for _, row in self.clinical_df.iterrows()
            }
            print(f"加载临床特征: {len(clinical_features)}个特征")

    def _filter_images_by_labels(self):
        """根据标签过滤图像"""
        original_count = len(self.image_files)
        available_labels = set(self.img_to_label.keys())

        self.image_files = [
            f for f in self.image_files
            if os.path.basename(f) in available_labels
        ]
        print(f"过滤后保留的图像数量: {len(self.image_files)} ({original_count - len(self.image_files)} 张被移除)")

        # 检查标签分布
        if self.image_files:
            labels = [self.img_to_label[os.path.basename(f)] for f in self.image_files]
            label_distribution = Counter(labels)
            print(f"{self.split}集过滤后的标签分布: {label_distribution}")

    def _check_label_mapping(self):
        """检查标签映射是否正确"""
        available_images = set(os.path.basename(f) for f in self.image_files)
        positive_images = [
            img for img, label in self.img_to_label.items()
            if label == 1 and img in available_images
        ]
        negative_images = [
            img for img, label in self.img_to_label.items()
            if label == 0 and img in available_images
        ]

        print(f"当前{self.split}集中可用的阳性样本: {len(positive_images)}")
        print(f"当前{self.split}集中可用的阴性样本: {len(negative_images)}")

        if len(positive_images) == 0:
            print("警告: 当前数据集中没有找到阳性样本！")
            print("可能的原因:")
            print("1. 标签映射逻辑不正确")
            print("2. 阳性样本没有被分配到当前split中")
            print("3. 图像文件名与标签文件中的Image Index不匹配")

            # 打印阳性样本示例
            positive_samples = self.labels_df[self.labels_df['tb_label'] == 1].sample(
                min(5, len(self.labels_df[self.labels_df['tb_label'] == 1])))
            print("\n阳性标签样本示例:")
            print(positive_samples[['Image Index', 'Finding Labels', 'tb_label']])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"索引 {idx} 超出范围，数据集大小为 {len(self)}")

        img_path = self.image_files[idx]
        img_name = os.path.basename(img_path)

        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"加载图像失败: {img_path}, 错误: {e}")
            return self[(idx + 1) % len(self)]  # 跳过错误图像

        # 应用肺部分割
        lung_mask = None
        if self.use_segmentation and self.segmentor:
            image, lung_mask = self._apply_segmentation(image)

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 获取标签
        label = self.img_to_label.get(img_name, 0)

        # 获取临床数据 (如果有)
        clinical_data = torch.zeros(0)  # 默认空张量
        if img_name in self.img_to_clinical:
            clinical_data = torch.from_numpy(self.img_to_clinical[img_name])

        return {
            'image': image,
            'label': label,
            'clinical': clinical_data,
            'mask': lung_mask if lung_mask is not None else torch.zeros(1),
            'path': img_path
        }

    def _apply_segmentation(self, image):
        """应用肺部分割"""
        original_size = image.size

        # 预处理图像用于分割模型
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(image).unsqueeze(0)

        # 分割预测
        self.segmentor.eval()
        with torch.no_grad():
            output = self.segmentor(input_tensor)['out'][0]

        mask = output.argmax(0).byte().cpu().numpy()
        mask_pil = Image.fromarray(mask * 255).resize(original_size, Image.NEAREST)

        # 应用掩码
        masked_image = Image.composite(
            image,
            Image.new('RGB', original_size, (0, 0, 0)),
            mask_pil
        )

        return masked_image, torch.from_numpy(np.array(mask_pil)).float().unsqueeze(0)


def load_segmentation_model(model_path=None):
    """加载改进的肺部分割模型"""
    model = deeplabv3_resnet50(weights=None, num_classes=2)

    if model_path and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"成功加载预训练分割模型: {model_path}")
        except Exception as e:
            print(f"加载预训练模型失败: {e}")

    return model.eval()


def create_datasets(data_dir, label_file=None, clinical_file=None,
                    use_segmentation=False, segmentor=None):
    """创建数据集"""
    # 创建变换
    train_transform = MedicalTransform(is_train=True)
    val_transform = MedicalTransform(is_train=False)

    # 创建数据集
    datasets = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = TBDataset(
            root_dir=data_dir,
            split=split,
            transform=train_transform if split == 'train' else val_transform,
            label_file=label_file,
            use_segmentation=use_segmentation,
            segmentor=segmentor,
            clinical_data_file=clinical_file if clinical_file else None
        )

    return datasets['train'], datasets['val'], datasets['test']


def create_data_loaders(train_dataset, val_dataset, test_dataset,
                        batch_size=32, num_workers=4):
    """创建优化的数据加载器"""

    def _create_weighted_sampler(dataset):
        """创建加权采样器处理类别不平衡"""
        labels = []
        for i in range(len(dataset)):
            labels.append(dataset[i]['label'])

        class_counts = Counter(labels)
        if len(class_counts) < 2:
            return None

        weights = 1. / torch.tensor([class_counts[l] for l in labels], dtype=torch.float)
        sampler = WeightedRandomSampler(weights, len(weights))
        return sampler

    # 训练集采样器
    train_sampler = _create_weighted_sampler(train_dataset)

    # 数据加载器
    loaders = {}
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    for split, dataset in [('val', val_dataset), ('test', test_dataset)]:
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    return loaders['train'], loaders['val'], loaders['test']


def visualize_samples(loader, title, save_dir='visualizations', num_samples=9):
    """改进的可视化函数"""
    os.makedirs(save_dir, exist_ok=True)

    try:
        batch = next(iter(loader))
        images = batch['image']
        labels = batch['label']
        paths = batch['path']
    except Exception as e:
        print(f"可视化失败: {e}")
        return

    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * std + mean

    # 创建可视化
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(min(num_samples, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f"标签: {labels[i].item()}\n{os.path.basename(paths[i])}")
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化结果保存至: {save_path}")


def analyze_dataset(dataset, split_name):
    """改进的数据集分析"""
    labels = []
    clinical_counts = []

    for i in tqdm(range(len(dataset)), desc=f"分析{split_name}"):
        sample = dataset[i]
        labels.append(sample['label'])
        if 'clinical' in sample and sample['clinical'].numel() > 0:
            clinical_counts.append(1)

    label_counts = Counter(labels)
    total = len(labels)

    print(f"\n{split_name} 分析结果:")
    print(f"总样本数: {total}")
    print("标签分布:")
    for label, count in label_counts.items():
        print(f"  类别 {label}: {count} ({count / total:.1%})")

    if clinical_counts:
        print(f"包含临床数据的样本: {len(clinical_counts)} ({len(clinical_counts) / total:.1%})")


def main():
    # 配置路径
    data_dir = r"D:\PycharmProjects\PythonProject1\TB_dataset"
    label_file = os.path.join(data_dir, "Data_Entry_2017.csv")
    clinical_file = os.path.join(data_dir, "clinical_data.csv") if os.path.exists(
        os.path.join(data_dir, "clinical_data.csv")) else None

    # 加载分割模型
    segmentor = load_segmentation_model() if False else None  # 暂时关闭分割

    # 创建数据集
    print("\n正在创建数据集...")
    train_dataset, val_dataset, test_dataset = create_datasets(
        data_dir, label_file, clinical_file, False, segmentor
    )

    # 分析数据集
    analyze_dataset(train_dataset, "训练集")
    analyze_dataset(val_dataset, "验证集")
    analyze_dataset(test_dataset, "测试集")

    # 创建数据加载器
    print("\n正在创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=16
    )

    # 可视化样本
    if len(train_loader.dataset) > 0:
        visualize_samples(train_loader, "训练集样本")
    if len(val_loader.dataset) > 0:
        visualize_samples(val_loader, "验证集样本")
    if len(test_loader.dataset) > 0:
        visualize_samples(test_loader, "测试集样本")


if __name__ == "__main__":
    main()