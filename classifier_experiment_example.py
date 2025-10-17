"""
分类器实验示例
演示如何使用数据集划分进行ResNet18分类实验
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision.models import resnet18
from tqdm import tqdm
import json
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import random

from load_dataset import MicroDopplerDataset


def set_random_seed(seed=42):
    """设置所有随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 已设置随机种子: {seed} (结果可重现)")


def train_classifier(model, train_loader, criterion, optimizer, device, epochs=15, scheduler=None):
    """训练分类器 - 固定epoch数，训练完成后再测试"""
    
    print(f"训练数据信息：{len(train_loader.dataset)} 张图像")
    
    # 检查第一个batch的图像尺寸
    for images, labels in train_loader:
        print(f"图像尺寸: {images.shape} (Batch, Channels, Height, Width)")
        print(f"图像数据类型: {images.dtype}, 值域: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    print(f"开始训练 {epochs} epochs（与文献一致）")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/(pbar.n+1):.4f}',
                'train_acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.2f}%")
        
        # 学习率调度（基于训练loss）
        if scheduler:
            scheduler.step(avg_train_loss)
    
    print(f"训练完成，共进行 {epochs} epochs（与文献一致）")


def extract_features(model, data_loader, device, max_samples=1000):
    """提取特征用于t-SNE可视化"""
    model.eval()
    
    # 创建特征提取器（去掉最后的分类层）
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    
    features = []
    labels = []
    
    sample_count = 0
    with torch.no_grad():
        for images, batch_labels in tqdm(data_loader, desc="Extracting features"):
            if sample_count >= max_samples:
                break
                
            images = images.to(device)
            
            # 提取特征
            batch_features = feature_extractor(images)  # [batch, 512, 1, 1]
            batch_features = batch_features.view(batch_features.size(0), -1)  # [batch, 512]
            
            features.append(batch_features.cpu().numpy())
            labels.extend(batch_labels.numpy())
            
            sample_count += len(batch_labels)
    
    features = np.concatenate(features, axis=0)[:max_samples]
    labels = np.array(labels)[:max_samples]
    
    return features, labels


def visualize_tsne_comparison(model, test_loader, device, per_class_accuracy):
    """对比可视化：准确率最高vs最低的用户，生成两张独立图片"""
    
    # 根据准确率排序用户
    user_accuracies = [(i, acc) for i, acc in enumerate(per_class_accuracy)]
    user_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    # 选择最高和最低准确率的用户
    top_5_users = [user_id for user_id, _ in user_accuracies[:5]]
    bottom_5_users = [user_id for user_id, _ in user_accuracies[-5:]]
    
    print(f"准确率最高的5个用户: {top_5_users}")
    print(f"对应准确率: {[user_accuracies[i][1] for i in range(5)]}")
    print(f"准确率最低的5个用户: {bottom_5_users}")  
    print(f"对应准确率: {[user_accuracies[i][1] for i in range(-5, 0)]}")
    
    # 深色颜色映射：红、黄、蓝、绿、紫
    colors = ['#CC0000', '#B8860B', '#000080', '#006400', '#4B0082']  # 深红、深黄、深蓝、深绿、深紫
    
    # === 第一张图：高准确率用户 ===
    features_high, labels_high = extract_features_for_users(
        model, test_loader, device, target_users=top_5_users, max_per_user=50
    )
    
    print("开始t-SNE降维（高准确率用户）...")
    tsne_high = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d_high = tsne_high.fit_transform(features_high)
    
    # 绘制高准确率用户
    plt.figure(figsize=(10, 8))
    for i, user_id in enumerate(top_5_users):
        mask = labels_high == user_id
        if mask.sum() > 0:
            plt.scatter(features_2d_high[mask, 0], features_2d_high[mask, 1], 
                       c=colors[i], label=f'User {user_id} ({user_accuracies[i][1]:.1f}%)', 
                       alpha=0.8, s=40, edgecolors='black', linewidth=0.8)
    
    plt.title('High Accuracy Users (Top 5)\nTest Set Feature Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsne_high_accuracy_users.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # === 第二张图：低准确率用户 ===
    features_low, labels_low = extract_features_for_users(
        model, test_loader, device, target_users=bottom_5_users, max_per_user=50
    )
    
    print("开始t-SNE降维（低准确率用户）...")
    tsne_low = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d_low = tsne_low.fit_transform(features_low)
    
    # 绘制低准确率用户
    plt.figure(figsize=(10, 8))
    for i, user_id in enumerate(bottom_5_users):
        mask = labels_low == user_id
        if mask.sum() > 0:
            acc_idx = len(user_accuracies) - 5 + i  # 计算在排序列表中的正确索引
            plt.scatter(features_2d_low[mask, 0], features_2d_low[mask, 1], 
                       c=colors[i], label=f'User {user_id} ({user_accuracies[acc_idx][1]:.1f}%)', 
                       alpha=0.8, s=40, edgecolors='black', linewidth=0.8)
    
    plt.title('Low Accuracy Users (Bottom 5)\nTest Set Feature Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsne_low_accuracy_users.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("高准确率用户t-SNE已保存至: tsne_high_accuracy_users.png")
    print("低准确率用户t-SNE已保存至: tsne_low_accuracy_users.png")


def extract_features_for_users(model, data_loader, device, target_users=None, max_per_user=50):
    """为特定用户提取特征"""
    model.eval()
    
    # 创建特征提取器
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    
    features = []
    labels = []
    user_counts = {user_id: 0 for user_id in target_users} if target_users else {}
    
    with torch.no_grad():
        for images, batch_labels in tqdm(data_loader, desc="Extracting specific user features"):
            images = images.to(device)
            
            for i, label in enumerate(batch_labels):
                user_id = label.item()
                
                # 只处理目标用户
                if target_users and user_id not in target_users:
                    continue
                    
                # 控制每个用户的样本数
                if target_users and user_counts[user_id] >= max_per_user:
                    continue
                
                # 提取单张图像的特征
                single_img = images[i:i+1]
                feature = feature_extractor(single_img)  # [1, 512, 1, 1]
                feature = feature.view(-1)  # [512]
                
                features.append(feature.cpu().numpy())
                labels.append(user_id)
                
                if target_users:
                    user_counts[user_id] += 1
    
    features = np.stack(features)
    labels = np.array(labels)
    
    print(f"提取了 {len(features)} 个特征，涉及用户: {np.unique(labels)}")
    return features, labels


def evaluate_classifier(model, test_loader, device, visualize=True):
    """评估分类器，包含过拟合检查和可视化"""
    model.eval()
    
    correct = 0
    total = 0
    per_class_correct = [0] * 31
    per_class_total = [0] * 31
    
    # 收集预测置信度分布
    all_confidences = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = probabilities.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 收集数据用于分析
            all_confidences.extend(confidences.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 统计每个类别
            for label, pred in zip(labels, predicted):
                per_class_total[label.item()] += 1
                if label == pred:
                    per_class_correct[label.item()] += 1
    
    overall_acc = 100. * correct / total
    
    print(f"\n整体准确率: {overall_acc:.2f}% ({correct}/{total})")
    
    # 分析置信度分布（过拟合检查）
    confidences = np.array(all_confidences)
    print(f"\n置信度分析（过拟合检查）:")
    print(f"  平均置信度: {confidences.mean():.3f}")
    print(f"  置信度标准差: {confidences.std():.3f}")
    print(f"  高置信度样本比例 (>0.9): {(confidences > 0.9).mean():.3f}")
    print(f"  低置信度样本比例 (<0.5): {(confidences < 0.5).mean():.3f}")
    
    # 如果置信度过于集中在高值，可能过拟合
    if confidences.mean() > 0.95:
        print("  ⚠️  警告: 平均置信度过高，可能存在过拟合！")
    if (confidences > 0.99).mean() > 0.5:
        print("  ⚠️  警告: 超过一半样本置信度>0.99，强烈怀疑过拟合！")
    
    # 每个用户的准确率
    print("\n各用户准确率:")
    for i in range(31):
        if per_class_total[i] > 0:
            acc = 100. * per_class_correct[i] / per_class_total[i]
            print(f"  用户{i:2d}: {acc:5.2f}% ({per_class_correct[i]}/{per_class_total[i]})")
    
    # 计算每个用户的准确率（百分比）
    per_class_accuracy = []
    for i in range(31):
        if per_class_total[i] > 0:
            acc = 100. * per_class_correct[i] / per_class_total[i]
            per_class_accuracy.append(acc)
        else:
            per_class_accuracy.append(0.0)
    
    # t-SNE对比可视化
    if visualize:
        print("\n进行t-SNE对比可视化（基于测试集）...")
        visualize_tsne_comparison(model, test_loader, device, per_class_accuracy)
    
    return overall_acc


class SyntheticDataset(Dataset):
    """
    加载生成的合成图像
    """
    def __init__(self, synthetic_folder, transform=None):
        self.samples = []
        
        synthetic_path = Path(synthetic_folder)
        
        # 搜索子文件夹中的图像：ID_X/user_Y_sample_Z.png
        for user_folder in sorted(synthetic_path.glob("ID_*")):
            if user_folder.is_dir():
                for img_path in sorted(user_folder.glob("user_*_sample_*.png")):
                    # 从文件名解析用户ID: user_Y_sample_Z.png
                    parts = img_path.stem.split('_')
                    if len(parts) >= 2 and parts[0] == 'user':
                        label = int(parts[1])
                        self.samples.append((img_path, label))
        
        self.transform = transform
        print(f"Loaded synthetic dataset: {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def main():
    """
    完整的分类器实验流程
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='ResNet18分类器实验')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--split_file', type=str,
                        default='./latents_cache/data_split.json',
                        help='数据集划分文件')
    parser.add_argument('--synthetic_folder', type=str, default=None,
                        help='合成数据文件夹（可选，用于增强实验）')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，确保结果可重现')
    args = parser.parse_args()
    
    # 设置随机种子 - 在所有操作之前
    set_random_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    print("="*60)
    print("加载数据集")
    print("="*60)
    
    # 训练集（真实图像）
    train_ds = MicroDopplerDataset(
        data_root=args.data_root,
        split_file=args.split_file,
        split='train',
        use_latents=False
    )
    
    # 如果提供了合成数据，添加到训练集
    if args.synthetic_folder:
        print("\n添加合成数据到训练集...")
        
        # 与真实图像使用相同的transform
        synthetic_ds = SyntheticDataset(
            args.synthetic_folder,
            transform=train_ds.transform
        )
        
        # 合并数据集
        train_ds = ConcatDataset([train_ds, synthetic_ds])
        print(f"增强后训练集: {len(train_ds)} 张（真实+合成）")
    
    # 测试集（真实图像）
    test_ds = MicroDopplerDataset(
        data_root=args.data_root,
        split_file=args.split_file,
        split='test',
        use_latents=False
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # 创建ResNet18分类器
    print("\n创建ResNet18分类器（不使用预训练）...")
    model = resnet18(pretrained=False, num_classes=31)
    model = model.to(device)
    
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # 学习率调度器：在训练loss停止下降时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 训练
    print("\n开始训练...")
    train_classifier(
        model, train_loader, criterion, optimizer, device, args.epochs, scheduler
    )
    
    # 训练完成后进行测试（包含t-SNE可视化）
    print("\n评估分类器...")
    accuracy = evaluate_classifier(model, test_loader, device, visualize=True)
    
    # 保存训练好的模型
    model_save_dir = Path("./trained_models")
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 根据是否使用合成数据生成不同的文件名
    if args.synthetic_folder:
        model_name = f"classifier_real_synthetic_acc{accuracy:.2f}_seed{args.seed}.pth"
        print(f"\n💾 保存增强模型: {model_name}")
    else:
        model_name = f"classifier_real_only_acc{accuracy:.2f}_seed{args.seed}.pth"
        print(f"\n💾 保存基线模型: {model_name}")
    
    model_save_path = model_save_dir / model_name
    
    # 保存完整的checkpoint信息
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'args': vars(args),
        'epoch': args.epochs,
        'model_info': {
            'architecture': 'ResNet18',
            'num_classes': 31,
            'input_size': (256, 256, 3),
            'data_type': 'real+synthetic' if args.synthetic_folder else 'real_only'
        }
    }
    
    torch.save(checkpoint, model_save_path)
    print(f"   模型已保存至: {model_save_path}")
    print(f"   测试准确率: {accuracy:.2f}%")
    if args.synthetic_folder:
        print(f"   数据类型: 真实图像 + 合成图像")
    else:
        print(f"   数据类型: 仅真实图像")
    
    print(f"\n🏆 训练完成！最终测试准确率: {accuracy:.2f}%")
    
    return accuracy


if __name__ == '__main__':
    accuracy = main()

