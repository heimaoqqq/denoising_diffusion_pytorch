"""
扩散模型生成图像质量评估脚本
==========================================
使用预训练的分类器评估DDPM生成图像的质量

评估指标：
1. 分类准确率 - 生成图像是否被正确分类
2. 置信度分布 - 分类器对生成图像的信心程度  
3. 每类别性能 - 哪些用户的生成效果更好
4. 真实度评估 - 与真实图像的对比分析
"""

import os
import sys
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

# 导入加载数据集模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_dataset import MicroDopplerDataset


class GeneratedImageDataset(Dataset):
    """加载生成图像数据集"""
    
    def __init__(self, generated_folder, transform=None):
        self.samples = []
        self.transform = transform
        
        generated_path = Path(generated_folder)
        print(f"扫描生成图像目录: {generated_path}")
        
        # 扫描生成图像：ID_X/user_Y_sample_Z.png
        for user_folder in sorted(generated_path.glob("ID_*")):
            if user_folder.is_dir():
                user_id = int(user_folder.name.split('_')[1])  # 提取用户ID
                label = user_id - 1  # 转换为0-30标签
                
                print(f"  处理文件夹: {user_folder.name}, 用户ID: {user_id}, 标签: {label}")
                
                for img_path in sorted(user_folder.glob("*.png")):
                    self.samples.append((img_path, label))
                    # 打印前几个文件名作为调试
                    if len(self.samples) <= 5:
                        print(f"    示例文件: {img_path.name} -> 标签: {label}")
        
        print(f"找到 {len(self.samples)} 张生成图像")
        
        # 统计每个用户的图像数量
        user_counts = {}
        for _, label in self.samples:
            user_counts[label] = user_counts.get(label, 0) + 1
        
        print("每用户生成图像数量:")
        for user_id in sorted(user_counts.keys()):
            print(f"  用户 {user_id:2d}: {user_counts[user_id]:3d} 张")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, label


def load_pretrained_classifier(model_path, device):
    """加载预训练的分类器"""
    print(f"加载预训练分类器: {model_path}")
    
    # 创建ResNet18模型
    model = resnet18(weights=None, num_classes=31)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  已加载checkpoint格式，epoch: {checkpoint.get('epoch', 'unknown')}")
        if 'accuracy' in checkpoint:
            print(f"  模型训练准确率: {checkpoint['accuracy']:.2f}%")
        if 'model_info' in checkpoint:
            print(f"  模型信息: {checkpoint['model_info']}")
    else:
        model.load_state_dict(checkpoint)
        print(f"  已加载state_dict格式")
    
    model = model.to(device)
    model.eval()
    print("分类器加载完成\n")
    
    return model


def evaluate_generated_images(model, generated_loader, device):
    """评估生成图像的质量"""
    print("开始评估生成图像...")
    
    model.eval()
    
    # 收集预测结果
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_probabilities = []
    
    correct = 0
    total = 0
    per_class_correct = [0] * 31
    per_class_total = [0] * 31
    
    batch_count = 0
    with torch.no_grad():
        for images, labels in tqdm(generated_loader, desc="评估中"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = probabilities.max(1)
            
            # 调试：打印前几个batch的预测结果
            if batch_count < 3:
                print(f"\n调试 - Batch {batch_count}:")
                for i in range(min(5, len(labels))):
                    print(f"  真实标签: {labels[i].item()}, 预测标签: {predicted[i].item()}, 置信度: {confidences[i].item():.3f}")
            batch_count += 1
            
            # 统计准确率
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 收集详细结果
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # 每类别统计
            for label, pred in zip(labels, predicted):
                per_class_total[label.item()] += 1
                if label == pred:
                    per_class_correct[label.item()] += 1
    
    # 转换为numpy数组便于分析
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    all_probabilities = np.array(all_probabilities)
    
    return {
        'overall_accuracy': 100. * correct / total,
        'total_samples': total,
        'correct_samples': correct,
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences,
        'probabilities': all_probabilities,
        'per_class_correct': per_class_correct,
        'per_class_total': per_class_total
    }


def analyze_results(results, save_dir):
    """分析评估结果并生成报告"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("生成图像质量评估报告")
    print(f"{'='*60}")
    
    # 1. 整体准确率
    overall_acc = results['overall_accuracy']
    total = results['total_samples']
    correct = results['correct_samples']
    
    print(f"\n📊 整体性能:")
    print(f"  总样本数: {total}")
    print(f"  正确分类: {correct}")
    print(f"  整体准确率: {overall_acc:.2f}%")
    
    # 2. 置信度分析
    confidences = results['confidences']
    print(f"\n🎯 置信度分析:")
    print(f"  平均置信度: {confidences.mean():.3f}")
    print(f"  置信度标准差: {confidences.std():.3f}")
    print(f"  高置信度样本 (>0.8): {(confidences > 0.8).mean()*100:.1f}%")
    print(f"  低置信度样本 (<0.5): {(confidences < 0.5).mean()*100:.1f}%")
    
    # 3. 每类别性能
    per_class_correct = results['per_class_correct']
    per_class_total = results['per_class_total']
    
    print(f"\n👥 各用户生成质量:")
    per_class_accuracies = []
    for i in range(31):
        if per_class_total[i] > 0:
            acc = 100. * per_class_correct[i] / per_class_total[i]
            per_class_accuracies.append(acc)
            print(f"  用户 {i:2d}: {acc:5.1f}% ({per_class_correct[i]:2d}/{per_class_total[i]:2d})")
        else:
            per_class_accuracies.append(0.0)
            print(f"  用户 {i:2d}: 无样本")
    
    # 4. 生成可视化
    create_visualizations(results, per_class_accuracies, save_dir)
    
    # 5. 质量评估
    print(f"\n🔍 质量评估:")
    if overall_acc > 80:
        print("  ✅ 优秀: 生成图像质量很高，能被准确分类")
    elif overall_acc > 60:
        print("  🟡 良好: 生成图像质量较好，但仍有改进空间")
    elif overall_acc > 40:
        print("  🟠 一般: 生成图像质量中等，需要调优")
    else:
        print("  ❌ 较差: 生成图像质量不佳，建议重新训练")
    
    if confidences.mean() > 0.7:
        print("  ✅ 分类器对生成图像有较高信心")
    else:
        print("  ⚠️ 分类器对生成图像信心不足，可能存在域间隙")


def create_visualizations(results, per_class_accuracies, save_dir):
    """创建可视化图表"""
    
    # 1. 置信度分布图
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(results['confidences'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(results['confidences'].mean(), color='red', linestyle='--', 
                label=f'Mean: {results["confidences"].mean():.3f}')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Generated Images\nConfidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 每类别准确率
    plt.subplot(1, 3, 2)
    users = list(range(31))
    bars = plt.bar(users, per_class_accuracies, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axhline(np.mean(per_class_accuracies), color='red', linestyle='--',
                label=f'Mean: {np.mean(per_class_accuracies):.1f}%')
    plt.xlabel('User ID')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-User Generation\nQuality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 标注特别好/差的用户
    mean_acc = np.mean(per_class_accuracies)
    for i, acc in enumerate(per_class_accuracies):
        if acc > mean_acc + 20:
            bars[i].set_color('darkgreen')
        elif acc < mean_acc - 20:
            bars[i].set_color('orange')
    
    # 3. 混淆矩阵（简化版 - 只显示对角线性能）
    plt.subplot(1, 3, 3)
    diagonal_performance = []
    for i in range(31):
        if results['per_class_total'][i] > 0:
            diagonal_performance.append(results['per_class_correct'][i] / results['per_class_total'][i])
        else:
            diagonal_performance.append(0)
    
    plt.plot(users, diagonal_performance, 'o-', color='purple', alpha=0.7)
    plt.axhline(np.mean(diagonal_performance), color='red', linestyle='--',
                label=f'Mean: {np.mean(diagonal_performance):.3f}')
    plt.xlabel('User ID')  
    plt.ylabel('Correct Classification Rate')
    plt.title('Classification Success\nRate per User')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'generation_quality_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  可视化图表已保存: {save_dir / 'generation_quality_analysis.png'}")
    plt.show()


def compare_with_real_images(model, real_loader, generated_loader, device, save_dir):
    """对比真实图像和生成图像的分类性能"""
    print("\n🔄 对比真实图像和生成图像...")
    
    # 评估真实图像
    print("评估真实图像...")
    real_results = evaluate_generated_images(model, real_loader, device)
    
    # 已经有生成图像结果
    gen_results = evaluate_generated_images(model, generated_loader, device)
    
    print(f"\n📊 对比结果:")
    print(f"  真实图像准确率: {real_results['overall_accuracy']:.2f}%")
    print(f"  生成图像准确率: {gen_results['overall_accuracy']:.2f}%")
    print(f"  差距: {real_results['overall_accuracy'] - gen_results['overall_accuracy']:.2f}%")
    
    print(f"\n  真实图像平均置信度: {real_results['confidences'].mean():.3f}")
    print(f"  生成图像平均置信度: {gen_results['confidences'].mean():.3f}")
    print(f"  差距: {real_results['confidences'].mean() - gen_results['confidences'].mean():.3f}")
    
    return real_results, gen_results


def main():
    parser = argparse.ArgumentParser(description='评估DDPM生成图像质量')
    
    # 路径参数
    parser.add_argument('--model_path', type=str,
                        default='/kaggle/working/denoising_diffusion_pytorch/trained_models/classifier_real_only_acc90.16_seed42.pth',
                        help='预训练分类器路径')
    parser.add_argument('--generated_folder', type=str,
                        default='/kaggle/input/generated',
                        help='生成图像文件夹')
    parser.add_argument('--real_data_root', type=str,
                        default='/kaggle/input/organized-gait-dataset/Normal_line',
                        help='真实数据根目录（用于对比）')
    parser.add_argument('--split_file', type=str,
                        default='./latents_cache/data_split.json',
                        help='数据划分文件')
    parser.add_argument('--save_dir', type=str,
                        default='./evaluation_results',
                        help='结果保存目录')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--compare_real', action='store_true',
                        help='是否与真实图像对比')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 图像预处理（与训练时保持一致）
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])
    
    # 1. 加载预训练分类器
    model = load_pretrained_classifier(args.model_path, device)
    
    # 2. 加载生成图像数据集
    print("加载生成图像...")
    generated_dataset = GeneratedImageDataset(args.generated_folder, transform=transform)
    generated_loader = DataLoader(
        generated_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 3. 评估生成图像
    results = evaluate_generated_images(model, generated_loader, device)
    
    # 4. 分析结果
    analyze_results(results, args.save_dir)
    
    # 5. 可选：与真实图像对比
    if args.compare_real:
        print("\n加载真实图像进行对比...")
        real_dataset = MicroDopplerDataset(
            data_root=args.real_data_root,
            split_file=args.split_file,
            split='test',  # 使用测试集对比
            use_latents=False,
            transform=transform
        )
        real_loader = DataLoader(
            real_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        compare_with_real_images(model, real_loader, generated_loader, device, args.save_dir)
    
    print(f"\n🎉 评估完成！结果保存在: {args.save_dir}")


if __name__ == '__main__':
    main()
