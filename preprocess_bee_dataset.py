import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import random
import math


def create_dirs(output_root):
    """创建输出目录结构（images和labels子文件夹）"""
    dirs = [
        os.path.join(output_root, 'images'),
        os.path.join(output_root, 'labels')
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return dirs


def is_valid_image(image_path):
    """检查图像是否有效（可打开且尺寸合理）"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        # 过滤过小的图像（蜜蜂目标可能不清晰）
        h, w = img.shape[:2]
        if h < 100 or w < 100:
            return False
        return True
    except:
        return False


def adaptive_histogram_equalization(img):
    """自适应直方图均衡化 - 改善光照不均匀"""
    # 转换到LAB色彩空间
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 对L通道应用自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # 合并通道
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def enhance_bee_features(img):
    """增强蜜蜂特征（黄黑条纹、翅膀透明度等）"""
    # 转换到HSV空间进行颜色增强
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 增强饱和度（突出黄黑条纹）
    s_enhanced = cv2.addWeighted(s, 1.4, np.zeros_like(s), 0, 10)
    s_enhanced = np.clip(s_enhanced, 0, 255)

    # 轻微增强亮度对比度
    v_enhanced = cv2.addWeighted(v, 1.1, np.zeros_like(v), 0, 5)
    v_enhanced = np.clip(v_enhanced, 0, 255)

    enhanced_hsv = cv2.merge([h, s_enhanced, v_enhanced])
    return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)


def denoise_image(img):
    """智能去噪 - 保留边缘细节的同时去除噪声"""
    # 使用双边滤波保留边缘
    denoised = cv2.bilateralFilter(img, 9, 75, 75)

    # 使用非局部均值去噪进一步优化
    denoised = cv2.fastNlMeansDenoisingColored(denoised, None, 10, 10, 7, 21)

    return denoised


def enhance_edges_preserve_color(img):
    """增强边缘特征同时保持颜色信息"""
    # 转换为灰度进行边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用Sobel算子检测边缘
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

    # 将边缘信息应用到每个颜色通道
    enhanced = img.copy().astype(np.float32)
    edge_weight = 0.15

    for i in range(3):  # BGR三个通道
        enhanced[:, :, i] = enhanced[:, :, i] + edge_weight * sobel_combined

    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced


def intelligent_background_suppression(img):
    """智能背景抑制 - 基于多特征的背景检测"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 创建多个背景掩码
    # 绿色植物背景
    lower_green1 = np.array([35, 30, 30])
    upper_green1 = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green1, upper_green1)

    # 棕色土壤/树干背景
    lower_brown = np.array([8, 50, 20])
    upper_brown = np.array([25, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # 蓝天背景
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # 合并背景掩码
    background_mask = cv2.bitwise_or(mask_green, cv2.bitwise_or(mask_brown, mask_blue))

    # 形态学操作优化掩码
    kernel = np.ones((5, 5), np.uint8)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)

    # 创建渐变抑制效果
    result = img.copy().astype(np.float32)
    background_suppression = 0.6  # 背景区域保留60%的亮度

    # 应用背景抑制
    mask_normalized = background_mask.astype(np.float32) / 255.0
    for i in range(3):
        result[:, :, i] = result[:, :, i] * (1 - mask_normalized * (1 - background_suppression))

    return result.astype(np.uint8)


def random_augmentation(img, apply_prob=0.5):
    """随机数据增强"""
    augmented = img.copy()

    # 随机亮度调整
    if random.random() < apply_prob:
        brightness = random.uniform(0.8, 1.2)
        augmented = cv2.convertScaleAbs(augmented, alpha=brightness, beta=0)

    # 随机对比度调整
    if random.random() < apply_prob:
        contrast = random.uniform(0.9, 1.1)
        augmented = cv2.convertScaleAbs(augmented, alpha=contrast, beta=0)

    # 随机高斯噪声（增强鲁棒性）
    if random.random() < 0.3:
        noise = np.random.normal(0, 8, augmented.shape).astype(np.uint8)
        augmented = cv2.add(augmented, noise)

    return augmented


def smart_resize(img, target_size=416):
    """智能缩放 - 保持比例并优化插值方法"""
    h, w = img.shape[:2]

    # 计算缩放比例
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # 根据缩放比例选择最优插值方法
    if scale < 0.5:
        # 大幅缩小：使用INTER_AREA
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    elif scale < 1.0:
        # 适度缩小：使用INTER_LINEAR
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        # 放大：使用INTER_CUBIC
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return resized, scale


def preprocess_bee_image(img, apply_augmentation=True, suppress_background=True):
    """完整的蜜蜂图像预处理流程"""
    original_shape = img.shape[:2]

    # 步骤1：去噪
    img = denoise_image(img)

    # 步骤2：光照归一化
    img = adaptive_histogram_equalization(img)

    # 步骤3：增强蜜蜂特征
    img = enhance_bee_features(img)

    # 步骤4：边缘增强（保持颜色）
    img = enhance_edges_preserve_color(img)

    # 步骤5：背景抑制
    if suppress_background:
        img = intelligent_background_suppression(img)

    # 步骤6：随机数据增强
    if apply_augmentation:
        img = random_augmentation(img)

    # 步骤7：智能缩放
    img, scale_factor = smart_resize(img, target_size=416)

    return img, scale_factor


def create_flipped_annotations(annotations, flip_type='horizontal'):
    """为翻转图像创建对应的标注"""
    flipped_annotations = []

    for ann in annotations:
        class_id, x_center, y_center, width, height = ann

        if flip_type == 'horizontal':
            # 水平翻转：x坐标翻转
            x_center = 1.0 - x_center
        elif flip_type == 'vertical':
            # 垂直翻转：y坐标翻转
            y_center = 1.0 - y_center

        flipped_annotations.append([class_id, x_center, y_center, width, height])

    return flipped_annotations


def augment_dataset(img, annotations, img_name, output_paths):
    """数据集扩充 - 生成翻转和旋转变体"""
    img_output_dir, lbl_output_dir = output_paths
    base_name = os.path.splitext(img_name)[0]
    ext = os.path.splitext(img_name)[1]

    augmented_images = []

    # 原图（已预处理）
    augmented_images.append((img, annotations, img_name))

    # 水平翻转
    flipped_h = cv2.flip(img, 1)
    flipped_h_annotations = create_flipped_annotations(annotations, 'horizontal')
    flipped_h_name = f"{base_name}_flip_h{ext}"
    augmented_images.append((flipped_h, flipped_h_annotations, flipped_h_name))

    # 可选：垂直翻转（蜜蜂检测中不太常用）
    if random.random() < 0.3:  # 30%概率添加垂直翻转
        flipped_v = cv2.flip(img, 0)
        flipped_v_annotations = create_flipped_annotations(annotations, 'vertical')
        flipped_v_name = f"{base_name}_flip_v{ext}"
        augmented_images.append((flipped_v, flipped_v_annotations, flipped_v_name))

    return augmented_images


def process_dataset(input_root, output_root, suppress_background=True, enable_augmentation=True):
    """批量处理数据集"""
    # 创建输出目录
    img_output_dir, lbl_output_dir = create_dirs(output_root)

    # 获取原始图像和标注文件路径
    img_input_dir = os.path.join(input_root, 'images')
    lbl_input_dir = os.path.join(input_root, 'labels')

    # 遍历所有图像文件
    valid_count = 0
    invalid_count = 0
    augmented_count = 0

    image_files = [f for f in os.listdir(img_input_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in tqdm(image_files, desc="预处理进度"):
        # 图像路径
        img_path = os.path.join(img_input_dir, img_name)
        # 对应标注文件路径
        lbl_name = os.path.splitext(img_name)[0] + '.txt'
        lbl_path = os.path.join(lbl_input_dir, lbl_name)

        # 检查图像有效性
        if not is_valid_image(img_path):
            invalid_count += 1
            continue

        # 读取图像
        img = cv2.imread(img_path)
        original_h, original_w = img.shape[:2]

        # 读取标注文件
        annotations = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = list(map(float, line.split()))
                    if len(parts) == 5:
                        annotations.append(parts)

        # 预处理图像
        processed_img, scale_factor = preprocess_bee_image(
            img,
            apply_augmentation=True,
            suppress_background=suppress_background
        )

        # 数据增强（可选）
        if enable_augmentation and annotations:  # 只对有标注的图像进行增强
            augmented_images = augment_dataset(
                processed_img,
                annotations,
                img_name,
                (img_output_dir, lbl_output_dir)
            )
        else:
            augmented_images = [(processed_img, annotations, img_name)]

        # 保存所有变体
        for aug_img, aug_annotations, aug_name in augmented_images:
            # 保存图像
            output_img_path = os.path.join(img_output_dir, aug_name)
            cv2.imwrite(output_img_path, aug_img)

            # 保存标注
            if aug_annotations:
                aug_lbl_name = os.path.splitext(aug_name)[0] + '.txt'
                with open(os.path.join(lbl_output_dir, aug_lbl_name), 'w') as f:
                    for ann in aug_annotations:
                        f.write(f"{int(ann[0])} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

            if aug_name != img_name:  # 统计增强的图像
                augmented_count += 1

        valid_count += 1

    # 打印处理结果
    print(f"预处理完成！")
    print(f"有效图像: {valid_count} 张")
    print(f"无效图像（已过滤）: {invalid_count} 张")
    print(f"数据增强生成: {augmented_count} 张")
    print(f"总输出图像: {valid_count + augmented_count} 张")
    print(f"处理后数据保存至: {output_root}")


if __name__ == "__main__":
    # 配置路径
    INPUT_DATASET_ROOT = r"E:\DIP_Bee_Detection\archive\train"
    OUTPUT_DATASET_ROOT = r"E:\DIP_Bee_Detection\archive\train_preprocessed"

    # 配置参数
    SUPPRESS_BACKGROUND = True  # 启用背景抑制
    ENABLE_AUGMENTATION = True  # 启用数据增强

    # 设置随机种子确保可重现性
    random.seed(42)
    np.random.seed(42)

    # 执行预处理
    process_dataset(
        INPUT_DATASET_ROOT,
        OUTPUT_DATASET_ROOT,
        suppress_background=SUPPRESS_BACKGROUND,
        enable_augmentation=ENABLE_AUGMENTATION
    )