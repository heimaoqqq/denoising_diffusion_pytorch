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

from load_dataset import MicroDopplerDataset


def train_classifier(model, train_loader, criterion, optimizer, device, epochs=100, scheduler=None):
    """训练分类器 - 基于训练loss早停，避免测试集泄露"""
    
    print(f"训练数据信息：{len(train_loader.dataset)} 张图像")
    
    # 检查第一个batch的图像尺寸
    for images, labels in train_loader:
        print(f"图像尺寸: {images.shape} (Batch, Channels, Height, Width)")
        print(f"图像数据类型: {images.dtype}, 值域: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    # 早停参数
    best_train_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 5  # 连续5个epoch训练loss不下降则停止
    
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
        
        # 计算epoch平均loss和准确率
        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.2f}%")
        
        # 基于训练loss的早停
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            patience_counter = 0
            print(f"  → 训练loss改善: {best_train_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  → 训练loss未改善 ({patience_counter}/{early_stop_patience})")
            
        # 早停检查
        if patience_counter >= early_stop_patience:
            print(f"\n训练loss连续 {early_stop_patience} epochs未改善，提前停止训练")
            print(f"最终训练loss: {best_train_loss:.4f}")
            break
            
        # 学习率调度（基于训练loss）
        if scheduler:
            scheduler.step(avg_train_loss)
    
    print(f"训练完成，共进行 {epoch+1} epochs")


def evaluate_classifier(model, test_loader, device):
    """评估分类器"""
    model.eval()
    
    correct = 0
    total = 0
    per_class_correct = [0] * 31
    per_class_total = [0] * 31
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
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
    
    print(f"\n整体准确率: {overall_acc:.2f}% ({correct}/{total})")
    
    # 每个用户的准确率
    print("\n各用户准确率:")
    for i in range(31):
        if per_class_total[i] > 0:
            acc = 100. * per_class_correct[i] / per_class_total[i]
            print(f"  用户{i:2d}: {acc:5.2f}% ({per_class_correct[i]}/{per_class_total[i]})")
    
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
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
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
    train_classifier(model, train_loader, criterion, optimizer, device, args.epochs, scheduler)
    
    # 评估
    print("\n评估分类器...")
    accuracy = evaluate_classifier(model, test_loader, device)
    
    return accuracy


if __name__ == '__main__':
    accuracy = main()

