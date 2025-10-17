"""
分类器实验 - 多次运行版本
基于classifier_experiment_example.py，增加多次运行功能
支持batch size对比和最低准确率模型选择
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
import argparse
import os

from load_dataset import MicroDopplerDataset


def train_classifier(model, train_loader, criterion, optimizer, device, epochs=15, scheduler=None):
    """训练分类器 - 固定epoch数，仿照文献做法，但监控训练状态"""
    
    # 检查第一个batch的图像尺寸
    for images, labels in train_loader:
        print(f"图像尺寸: {images.shape} (Batch, Channels, Height, Width)")
        print(f"图像数据类型: {images.dtype}, 值域: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    print(f"开始训练 {epochs} epochs（仿照文献设置）")
    
    # 记录训练历史用于判断是否收敛
    train_history = {
        'losses': [],
        'accuracies': []
    }
    
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
        
        # 记录训练历史
        train_history['losses'].append(avg_train_loss)
        train_history['accuracies'].append(train_acc)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.2f}%")
        
        # 学习率调度（基于训练loss）
        if scheduler:
            scheduler.step(avg_train_loss)
    
    # 分析训练是否充分
    analyze_training_convergence(train_history, epochs)
    
    print(f"训练完成，共进行 {epochs} epochs（与文献一致）")


def analyze_training_convergence(train_history, epochs):
    """分析训练是否收敛，判断epoch数是否充分"""
    
    losses = train_history['losses']
    accuracies = train_history['accuracies']
    
    # 检查最后几个epoch的改善情况
    if len(losses) >= 5:
        # 计算最后5个epoch的平均改善
        recent_loss_trend = np.mean(np.diff(losses[-5:]))
        recent_acc_trend = np.mean(np.diff(accuracies[-5:]))
        
        print(f"\n📈 训练收敛分析:")
        print(f"最后5个epoch loss变化: {recent_loss_trend:+.6f}")
        print(f"最后5个epoch准确率变化: {recent_acc_trend:+.2f}%")
        
        # 判断是否还在改善
        if abs(recent_loss_trend) > 0.01:  # loss还在显著下降
            print("⚠️  警告: Loss还在显著下降，可能需要更多epoch")
        elif recent_acc_trend > 1.0:  # 准确率还在显著提升
            print("⚠️  警告: 准确率还在显著提升，可能需要更多epoch")
        else:
            print("✅ 训练基本收敛，15 epoch应该足够")
            
        # 检查是否可能欠拟合
        final_acc = accuracies[-1]
        if final_acc < 60:
            print("❌ 最终准确率过低，强烈建议增加epoch数或调整学习率")
        elif final_acc < 70:
            print("⚠️  最终准确率较低，考虑增加epoch数")


def evaluate_classifier(model, test_loader, device, verbose=True):
    """评估分类器 - 简化版，只返回准确率"""
    model.eval()
    
    correct = 0
    total = 0
    per_class_correct = [0] * 31
    per_class_total = [0] * 31
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", disable=not verbose):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 统计每个类别
            for label, pred in zip(labels, predicted):
                per_class_total[label.item()] += 1
                if label == pred:
                    per_class_correct[label.item()] += 1
    
    overall_acc = 100. * correct / total
    
    if verbose:
        print(f"\n整体准确率: {overall_acc:.2f}% ({correct}/{total})")
        
        # 每个用户的准确率
        print("\n各用户准确率:")
        for i in range(31):
            if per_class_total[i] > 0:
                acc = 100. * per_class_correct[i] / per_class_total[i]
                print(f"  用户{i:2d}: {acc:5.2f}% ({per_class_correct[i]}/{per_class_total[i]})")
    
    return overall_acc, per_class_correct, per_class_total


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
    
    print("Starting t-SNE dimensionality reduction (high accuracy users)...")
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
    
    print("Starting t-SNE dimensionality reduction (low accuracy users)...")
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
    
    print("High accuracy users t-SNE saved to: tsne_high_accuracy_users.png")
    print("Low accuracy users t-SNE saved to: tsne_low_accuracy_users.png")


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
    
    print(f"Extracted {len(features)} features from users: {np.unique(labels)}")
    return features, labels


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


def run_single_experiment(data_root, split_file, synthetic_folder, batch_size, epochs, lr, device, seed=None, verbose=False):
    """运行单次实验"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    train_ds = MicroDopplerDataset(
        data_root=data_root,
        split_file=split_file,
        split='train',
        use_latents=False
    )
    
    # 如果提供了合成数据，添加到训练集
    if synthetic_folder:
        synthetic_ds = SyntheticDataset(
            synthetic_folder,
            transform=train_ds.transform
        )
        train_ds = ConcatDataset([train_ds, synthetic_ds])
    
    # 测试集（真实图像）
    test_ds = MicroDopplerDataset(
        data_root=data_root,
        split_file=split_file,
        split='test',
        use_latents=False
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # 创建ResNet18分类器
    model = resnet18(weights=None, num_classes=31)
    model = model.to(device)
    
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # 训练
    if verbose:
        print(f"\n训练数据信息：{len(train_loader.dataset)} 张图像")
        print("开始训练...")
    train_classifier(model, train_loader, criterion, optimizer, device, epochs, scheduler)
    
    # 评估
    if verbose:
        print("评估分类器...")
    accuracy, per_class_correct, per_class_total = evaluate_classifier(model, test_loader, device, verbose=verbose)
    
    return accuracy, model, test_loader, device, per_class_correct, per_class_total


def run_multiple_experiments(data_root, split_file, synthetic_folder, batch_sizes, epochs, lr, device, num_runs=10):
    """运行多次实验，对比不同batch size"""
    
    results = {}
    all_saved_models = []  # 记录所有保存的模型信息
    
    for batch_size in batch_sizes:
        print(f"\n🎯 测试 Batch Size = {batch_size}")
        print("="*70)
        
        batch_results = []
        batch_models = []
        
        for run in range(num_runs):
            print(f"\n📊 第 {run+1}/{num_runs} 次运行 (Batch Size = {batch_size})")
            
            # 使用不同的随机种子
            accuracy, model, test_loader, device_used, per_class_correct, per_class_total = run_single_experiment(
                data_root=data_root,
                split_file=split_file, 
                synthetic_folder=synthetic_folder,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                device=device,
                seed=42 + run,
                verbose=True
            )
            
            batch_results.append(accuracy)
            batch_models.append((accuracy, model, test_loader, device_used, per_class_correct, per_class_total))
            
            print(f"✅ 准确率: {accuracy:.2f}%")
            
            # 保存每一个模型
            model_path = f"model_batch{batch_size}_run{run+1}_acc{accuracy:.2f}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"💾 模型已保存: {model_path}")
            
            # 记录模型信息
            model_info = {
                'batch_size': batch_size,
                'run': run + 1,
                'accuracy': accuracy,
                'path': model_path
            }
            all_saved_models.append(model_info)
        
        # 统计结果
        batch_results = np.array(batch_results)
        print(f"\n📈 Batch Size {batch_size} 统计结果:")
        print(f"所有结果: {[f'{acc:.2f}%' for acc in batch_results]}")
        print(f"平均准确率: {batch_results.mean():.2f}%")
        print(f"标准差: {batch_results.std():.2f}%")
        print(f"最高准确率: {batch_results.max():.2f}%")
        print(f"最低准确率: {batch_results.min():.2f}%")
        
        # 找到最低准确率的模型
        min_idx = np.argmin(batch_results)
        min_accuracy = batch_results[min_idx]
        min_model_info = batch_models[min_idx]
        
        print(f"🎯 选择最低准确率模型: 第 {min_idx+1} 次运行, 准确率: {min_accuracy:.2f}%")
        
        results[batch_size] = {
            'all_results': batch_results.tolist(),
            'mean': batch_results.mean(),
            'std': batch_results.std(),
            'max': batch_results.max(),
            'min': batch_results.min(),
            'min_model_info': min_model_info,
            'min_run_idx': min_idx + 1
        }
    
    # 返回结果包含所有模型信息
    return {
        'batch_results': results,
        'all_saved_models': all_saved_models
    }


def generate_detailed_analysis(results, batch_sizes):
    """生成详细分析和可视化"""
    
    print("\n" + "="*80)
    print("🏆 最终结果对比分析")
    print("="*80)
    
    # 获取批次结果和所有模型信息
    batch_results = results['batch_results']
    all_saved_models = results['all_saved_models']
    
    # 对比表格
    print(f"\n📊 Batch Size 对比:")
    print(f"{'Batch Size':<12}{'平均准确率':<12}{'标准差':<10}{'最高准确率':<12}{'最低准确率':<12}")
    print("-" * 70)
    
    for batch_size in batch_sizes:
        result = batch_results[batch_size]
        print(f"{batch_size:<12}{result['mean']:<12.2f}{result['std']:<10.2f}{result['max']:<12.2f}{result['min']:<12.2f}")
    
    # 显示所有保存的模型信息
    print(f"\n💾 所有保存的模型 ({len(all_saved_models)} 个):")
    print("-" * 70)
    for i, model_info in enumerate(all_saved_models):
        print(f"{i+1:2d}. {model_info['path']}")
        print(f"    Batch Size: {model_info['batch_size']}, Run: {model_info['run']}, 准确率: {model_info['accuracy']:.2f}%")
    
    print(f"\n📋 模型选择建议:")
    print(f"所有准确率范围: {min([m['accuracy'] for m in all_saved_models]):.2f}% - {max([m['accuracy'] for m in all_saved_models]):.2f}%")
    print(f"平均准确率: {np.mean([m['accuracy'] for m in all_saved_models]):.2f}%")
    print(f"你可以根据需要手动选择任何一个模型作为基准")
    
    return {
        'batch_results': batch_results,
        'all_saved_models': all_saved_models,
        'all_results': results
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ResNet18分类器多次运行实验')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--split_file', type=str,
                        default='./latents_cache/data_split.json',
                        help='数据集划分文件')
    parser.add_argument('--synthetic_folder', type=str, default=None,
                        help='合成数据文件夹（可选，用于增强实验）')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[8, 16],
                        help='要测试的batch size列表')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='每个batch size的运行次数')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--adaptive_epochs', action='store_true',
                        help='如果训练未收敛，自动延长到最多30 epochs')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print("🚀 开始多次运行分类器实验")
    print(f"测试Batch Sizes: {args.batch_sizes}")
    print(f"每个batch size运行次数: {args.num_runs}")
    print(f"总实验次数: {len(args.batch_sizes) * args.num_runs}")
    print("="*80)
    
    # 运行多次实验
    results = run_multiple_experiments(
        data_root=args.data_root,
        split_file=args.split_file,
        synthetic_folder=args.synthetic_folder,
        batch_sizes=args.batch_sizes,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        num_runs=args.num_runs
    )
    
    # 生成详细分析
    final_results = generate_detailed_analysis(results, args.batch_sizes)
    
    return final_results


if __name__ == '__main__':
    results = main()
    print(f"\n🏆 实验完成!")
    print(f"💾 总共保存了 {len(results['all_saved_models'])} 个模型")
    print("   你可以手动选择任何一个作为DDPM对比实验的基准")
