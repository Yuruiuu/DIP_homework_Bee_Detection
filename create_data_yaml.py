import os


def create_bee_data_yaml(dataset_root, output_yaml_path):
    """
    创建蜜蜂数据集的配置YAML文件，包含train和val字段
    :param dataset_root: 数据集根目录（预处理后的根目录）
    :param output_yaml_path: 输出YAML文件路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_yaml_path)
    os.makedirs(output_dir, exist_ok=True)

    # YAML内容（按要求格式生成）
    yaml_content = f"""path: {dataset_root}  # 改为预处理后的根目录
train: images  # 训练集图像文件夹
val: images    # 验证集图像文件夹（若有独立验证集，需修改路径）
labels: labels  # 标注文件夹（train和val共享的标注目录）
nc: 1  # 类别数量
names: ['bees']  # 类别名称
"""

    # 写入文件
    with open(output_yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"数据集配置文件已生成: {output_yaml_path}")


if __name__ == "__main__":
    # 配置路径（请修改为你的实际路径）
    # 注意：此处应为预处理后的根目录，而非labels子目录
    DATASET_ROOT = r"E:\DIP_Bee_Detection\archive\valid"  # 数据集根目录
    OUTPUT_YAML_PATH = r"E:\DIP_Bee_Detection\archive\valid\bee_data.yaml"  # 输出YAML文件路径

    # 生成配置文件
    create_bee_data_yaml(DATASET_ROOT, OUTPUT_YAML_PATH)
