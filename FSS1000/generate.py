# only trn and val is used in fss training, so pretrain can only use trn

import os
import random
from typing import List, Tuple


def read_classes(file_path: str) -> List[str]:
    """读取类别文件，返回类别列表"""
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"错误: 未找到类别文件 {file_path}")
        return []

def find_image_mask_pairs(data_root: str, classes: List[str]) -> List[Tuple[str, str]]:
    """查找并返回所有图像-掩码对"""
    pairs = []
    
    for cls in classes:
        cls_dir = os.path.join(data_root, cls)
        if not os.path.isdir(cls_dir):
            print(f"警告: 类别目录 {cls_dir} 不存在，跳过")
            continue
        
        # 构建图像和掩码的映射
        image_files = {}
        mask_files = {}
        
        for file in os.listdir(cls_dir):
            file_path = os.path.join(cls_dir, file)
            if not os.path.isfile(file_path):
                continue
                
            base_name, ext = os.path.splitext(file)
            if ext.lower() == '.jpg':
                image_files[base_name] = file
            elif ext.lower() == '.png':
                mask_files[base_name] = file
        
        # 匹配图像和掩码
        for name in image_files:
            if name in mask_files:
                pairs.append((
                    os.path.join(cls, image_files[name]),
                    os.path.join(cls, mask_files[name])
                ))
    
    return pairs

def split_dataset(pairs: List[Tuple[str, str]], train_ratio: float = 0.8) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """将数据集按比例划分为训练集和测试集"""
    random.shuffle(pairs)
    split_index = int(len(pairs) * train_ratio)
    return pairs[:split_index], pairs[split_index:]

if __name__ == "__main__":
    # 配置参数
    train_file = "/share/home/orfu/DeepLearning/MyModels/CAM_BAM_Pretrain/lists/FSS1000/splits/trn.txt"         # 类别文件路径
    data_root = "/share/home/orfu/DeepLearning/Dataset/PrivateDataset/FSS-Datasets/FSS-1000"                    # 数据根目录
    output_train = "/share/home/orfu/DeepLearning/MyModels/CAM_BAM_Pretrain/lists/FSS1000/trn.txt"              # 训练集输出文件
    output_test = "/share/home/orfu/DeepLearning/MyModels/CAM_BAM_Pretrain/lists/FSS1000/val.txt"               # 测试集输出文件
    
    # 读取类别
    classes = read_classes(train_file)
    if not classes:
        print("没有找到任何类别，程序退出")
        exit()
    
    # 查找图像-掩码对
    pairs = find_image_mask_pairs(data_root, classes)
    if not pairs:
        print("没有找到任何图像-掩码对，程序退出")
        exit()
    
    print(f"共找到 {len(pairs)} 对图像-掩码数据")
    
    # 划分数据集
    train_pairs, test_pairs = split_dataset(pairs, 0.8)
    
    print(f"训练集: {len(train_pairs)} 对")
    print(f"测试集: {len(test_pairs)} 对")
    
    # 保存到文件
    with open(output_train, 'w') as f:
        for img, mask in train_pairs:
            f.write(f"{img} {mask}\n")

    with open(output_test, 'w') as f:
        for img, mask in test_pairs:
            f.write(f"{img} {mask}\n")
    
    print(f"训练集已保存到 {output_train}")
    print(f"测试集已保存到 {output_test}")