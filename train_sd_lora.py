"""
SD+LoRA训练脚本 - 直接使用Diffusers API
不依赖外部训练脚本，完全自包含
"""

import sys
import os
from pathlib import Path
import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
import numpy as np

from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator


class TextImageDataset(Dataset):
    """文本-图像配对数据集"""
    
    def __init__(self, data_root, tokenizer, resolution=512):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.resolution = resolution
        
        # 读取metadata.jsonl
        metadata_file = self.data_root / "metadata.jsonl"
        self.data = []
        with open(metadata_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图像
        image_path = self.data_root / "images" / item["file_name"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        
        # Tokenize文本
        text = item["text"]
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids[0],
        }


def train_sd_lora(
    model_name="runwayml/stable-diffusion-v1-5",
    dataset_path="./sd_lora_dataset",
    val_dataset_path=None,
    output_dir="./sd_lora_output",
    resolution=512,
    train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=100,
    learning_rate=1e-4,
    lr_scheduler="constant",
    lr_warmup_steps=0,
    seed=42,
    lora_rank=8,
    lora_alpha=8,
    validation_prompt="user 0",
    validation_epochs=10,
    num_validation_images=4,
    checkpointing_steps=500,
    mixed_precision="fp16",
    gradient_checkpointing=True,
    use_8bit_adam=False,
    center_crop=True,
    random_flip=True,
    report_to="tensorboard"
):
    """
    训练SD+LoRA模型
    
    Args:
        model_name: 预训练模型名称
        dataset_path: 训练集路径（包含images/和metadata.jsonl）
        val_dataset_path: 验证集路径（可选，如果为None则只用validation_prompt）
        output_dir: 输出目录
        resolution: 图像分辨率
        train_batch_size: 训练batch size
        gradient_accumulation_steps: 梯度累积步数
        num_train_epochs: 训练epoch数
        learning_rate: 学习率
        lr_scheduler: 学习率调度器
        lr_warmup_steps: 预热步数
        seed: 随机种子
        lora_rank: LoRA秩
        lora_alpha: LoRA缩放因子
        validation_prompt: 验证提示词
        validation_epochs: 验证频率（epoch）
        num_validation_images: 每次验证生成的图像数
        checkpointing_steps: 保存检查点频率（步数）
        mixed_precision: 混合精度训练
        gradient_checkpointing: 是否使用梯度检查点
        use_8bit_adam: 是否使用8bit Adam
        center_crop: 是否中心裁剪
        random_flip: 是否随机翻转
        report_to: 日志记录工具
    """
    
    print("="*60)
    print("SD+LoRA训练 - 微多普勒时频图生成")
    print("="*60)
    print()
    print("配置:")
    print(f"  模型: {model_name}")
    print(f"  训练集: {dataset_path}")
    if val_dataset_path:
        print(f"  验证集: {val_dataset_path}")
    print(f"  输出: {output_dir}")
    print(f"  分辨率: {resolution}x{resolution}")
    print(f"  Batch size: {train_batch_size} (有效: {train_batch_size * gradient_accumulation_steps})")
    print(f"  Epochs: {num_train_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  LoRA alpha: {lora_alpha}")
    print(f"  验证频率: 每{validation_epochs}个epoch")
    print()
    
    # 检查数据集是否存在
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        print("请先运行: python prepare_sd_lora_dataset.py")
        sys.exit(1)
    
    metadata_file = dataset_path / "metadata.jsonl"
    if not metadata_file.exists():
        print(f"错误: metadata.jsonl不存在: {metadata_file}")
        print("请先运行: python prepare_sd_lora_dataset.py")
        sys.exit(1)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    
    # 设置随机种子
    torch.manual_seed(seed)
    
    print("="*60)
    print("开始训练...")
    print("="*60)
    print()
    
    # 1. 加载模型
    print("1. 加载预训练模型...")
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    
    # 冻结VAE和text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # 2. 配置LoRA
    print("2. 配置LoRA...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # 3. 创建数据集和DataLoader
    print("3. 加载数据集...")
    train_dataset = TextImageDataset(dataset_path, tokenizer, resolution)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # 4. 创建优化器
    print("4. 创建优化器...")
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except:
            print("  Warning: bitsandbytes未安装，使用标准AdamW")
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
    )
    
    # 5. 创建学习率调度器
    lr_scheduler_obj = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=len(train_dataloader) * num_train_epochs,
    )
    
    # 6. 使用Accelerator准备
    unet, optimizer, train_dataloader, lr_scheduler_obj = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler_obj
    )
    
    # 将VAE和text_encoder移到设备
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # 7. 训练循环
    print("5. 开始训练...")
    print(f"  总步数: {len(train_dataloader) * num_train_epochs}")
    print(f"  每epoch步数: {len(train_dataloader)}")
    print()
    
    global_step = 0
    for epoch in range(num_train_epochs):
        unet.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_train_epochs}")
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet):
                # 编码图像到潜在空间
                latents = vae.encode(batch["pixel_values"].to(accelerator.device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # 采样噪声
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                
                # 添加噪声
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 获取文本嵌入
                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]
                
                # 预测噪声
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # 计算损失
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # 反向传播
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler_obj.step()
                optimizer.zero_grad()
            
            # 更新进度条
            progress_bar.set_postfix({"loss": loss.detach().item()})
            global_step += 1
            
            # 保存检查点
            if global_step % checkpointing_steps == 0:
                if accelerator.is_main_process:
                    save_path = output_path / f"checkpoint-{global_step}"
                    save_path.mkdir(exist_ok=True)
                    unet.save_pretrained(save_path)
        
        # 每个epoch结束后的验证（每轮都验证）
        if accelerator.is_main_process:
            print(f"\n验证 Epoch {epoch+1}...")
            
            # 生成验证图像
            unet.eval()
            validation_dir = output_path / "validation_images"
            validation_dir.mkdir(exist_ok=True)
            
            # 创建pipeline用于生成（使用DPM-Solver++调度器）
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
            
            # 配置DPM-Solver++调度器（20步即可，质量接近DDIM 100步）
            # 可选：DDIMScheduler（100步，最高质量但慢）
            dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_name, subfolder="scheduler")
            
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                unet=accelerator.unwrap_model(unet),
                text_encoder=text_encoder,
                vae=vae,
                tokenizer=tokenizer,
                scheduler=dpm_scheduler,
                safety_checker=None,
                torch_dtype=torch.float16 if mixed_precision == "fp16" else torch.float32,
            )
            pipeline = pipeline.to(accelerator.device)
            
            # 生成图像（DPM-Solver++ 20步，条件扩散）
            for i in range(num_validation_images):
                with torch.no_grad():
                    # 生成512×512图像
                    image_512 = pipeline(
                        validation_prompt,  # 条件文本："user 0"
                        num_inference_steps=20,  # DPM-Solver++ 20步（质量接近DDIM 100步）
                        guidance_scale=7.5,  # CFG强度
                    ).images[0]
                    
                    # Resize到256×256（与训练数据一致）
                    image_256 = image_512.resize((256, 256), Image.LANCZOS)
                    
                    # 保存图像
                    image_path = validation_dir / f"epoch_{epoch+1:03d}_sample_{i}.png"
                    image_256.save(image_path)
            
            print(f"  ✓ 验证图像已保存到: {validation_dir} (256×256, DPM-Solver++ 20步)")
            
            # 清理pipeline释放显存
            del pipeline
            del dpm_scheduler
            torch.cuda.empty_cache()
            unet.train()
    
    # 8. 保存最终模型
    print("\n6. 保存最终模型...")
    if accelerator.is_main_process:
        unet.save_pretrained(output_path)
        print(f"  ✓ 模型已保存到: {output_path}")
    
    print()
    print("="*60)
    print("训练完成！")
    print("="*60)
    print(f"输出目录: {output_dir}")
    print(f"LoRA权重: {output_dir}")
    print()
    print("生成图像:")
    print(f"  python generate_sd_lora.py --lora_weights {output_dir} --user_id 0")
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(description='SD+LoRA训练 - Python版本')
    
    # 路径参数
    parser.add_argument('--model_name', type=str,
                        default='runwayml/stable-diffusion-v1-5',
                        help='预训练模型名称')
    parser.add_argument('--dataset_path', type=str,
                        default='./sd_lora_dataset',
                        help='训练集路径')
    parser.add_argument('--val_dataset_path', type=str,
                        default=None,
                        help='验证集路径（可选）')
    parser.add_argument('--output_dir', type=str,
                        default='./sd_lora_output',
                        help='输出目录')
    
    # 训练参数
    parser.add_argument('--resolution', type=int, default=512,
                        help='图像分辨率')
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='训练batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='梯度累积步数')
    parser.add_argument('--num_train_epochs', type=int, default=100,
                        help='训练epoch数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--lr_scheduler', type=str, default='constant',
                        choices=['linear', 'cosine', 'cosine_with_restarts', 
                                'polynomial', 'constant', 'constant_with_warmup'],
                        help='学习率调度器')
    parser.add_argument('--lr_warmup_steps', type=int, default=0,
                        help='学习率预热步数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    # LoRA参数
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA秩（越大表达能力越强，推荐4-16）')
    parser.add_argument('--lora_alpha', type=int, default=8,
                        help='LoRA缩放因子（通常等于rank）')
    
    # 验证和保存
    parser.add_argument('--validation_prompt', type=str, default='user 0',
                        help='验证提示词')
    parser.add_argument('--validation_epochs', type=int, default=5,
                        help='每N个epoch验证一次（默认5）')
    parser.add_argument('--num_validation_images', type=int, default=4,
                        help='每次验证生成的图像数')
    parser.add_argument('--checkpointing_steps', type=int, default=500,
                        help='每N步保存一次检查点')
    
    # 优化参数
    parser.add_argument('--mixed_precision', type=str, default='fp16',
                        choices=['no', 'fp16', 'bf16'],
                        help='混合精度训练')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                        help='使用梯度检查点节省显存')
    parser.add_argument('--no_gradient_checkpointing', dest='gradient_checkpointing',
                        action='store_false',
                        help='不使用梯度检查点')
    parser.add_argument('--use_8bit_adam', action='store_true', default=False,
                        help='使用8bit Adam优化器（需要bitsandbytes）')
    
    # 数据增强
    parser.add_argument('--center_crop', action='store_true', default=True,
                        help='中心裁剪')
    parser.add_argument('--no_center_crop', dest='center_crop', action='store_false',
                        help='不使用中心裁剪')
    parser.add_argument('--random_flip', action='store_true', default=True,
                        help='随机水平翻转')
    parser.add_argument('--no_random_flip', dest='random_flip', action='store_false',
                        help='不使用随机翻转')
    
    # 日志
    parser.add_argument('--report_to', type=str, default='tensorboard',
                        choices=['tensorboard', 'wandb', 'all'],
                        help='日志记录工具')
    
    args = parser.parse_args()
    
    # 运行训练
    success = train_sd_lora(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        val_dataset_path=args.val_dataset_path,
        output_dir=args.output_dir,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        seed=args.seed,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        validation_prompt=args.validation_prompt,
        validation_epochs=args.validation_epochs,
        num_validation_images=args.num_validation_images,
        checkpointing_steps=args.checkpointing_steps,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        use_8bit_adam=args.use_8bit_adam,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        report_to=args.report_to
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
