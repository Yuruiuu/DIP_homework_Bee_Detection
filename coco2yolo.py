# 将COCO JSON标注转换为YOLO格式
import json
import os
from pathlib import Path


def coco_to_yolo(coco_json_path, output_labels_dir, img_width=416, img_height=416):
    """
    转换COCO格式标注为YOLO格式
    :param coco_json_path: COCO JSON标注文件路径
    :param output_labels_dir: YOLO格式标注输出目录
    :param img_width: 图像宽度
    :param img_height: 图像高度
    """
    # 创建输出目录
    os.makedirs(output_labels_dir, exist_ok=True)

    # 加载COCO JSON文件
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 创建图像ID到文件名的映射
    img_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

    # 处理每个标注
    for annotation in coco_data['annotations']:
        img_id = annotation['image_id']
        img_filename = img_id_to_filename[img_id]

        # 获取YOLO标注文件名（与图像同名，改为txt后缀）
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(output_labels_dir, label_filename)

        # 提取边界框信息 (COCO格式: xmin, ymin, width, height)
        xmin, ymin, width, height = annotation['bbox']

        # 转换为YOLO格式 (归一化的x_center, y_center, width, height)
        x_center = (xmin + width / 2) / img_width
        y_center = (ymin + height / 2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height

        # 合并重复的bees类别（原数据集中有id=0和id=1两个bees类别）
        class_id = 0  # 统一为0类

        # 写入标注文件
        with open(label_path, 'a') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

    print(f"转换完成! 共处理 {len(coco_data['images'])} 张图像，标注文件保存至 {output_labels_dir}")


if __name__ == "__main__":
    # 配置路径（请修改为你的实际路径）
    COCO_JSON_PATH = r"E:\DIP_Bee_Detection\archive\valid\_annotations.coco.json"  # 你的COCO JSON文件路径
    OUTPUT_LABELS_DIR = r"E:\DIP_Bee_Detection\archive\valid\labels"  # 输出YOLO格式标注的目录

    # 执行转换
    coco_to_yolo(COCO_JSON_PATH, OUTPUT_LABELS_DIR)
