# 第一步：解决OpenMP冲突（与评估脚本一致，避免训练中报错）
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import subprocess
import sys


def train_yolov5_bee_model(
        data_yaml_path,
        model_config_path,
        epochs=100,
        img_size=416,
        batch_size=16,
        device='0',
        project_dir="bee_training_results",
        experiment_name="yolov5m_bee_train"
):
    """
    使用蜜蜂数据集训练YOLOv5模型
    :param data_yaml_path: 数据集配置YAML路径（bee_data.yaml）
    :param model_config_path: YOLOv5模型配置文件路径（如yolov5m.yaml，在YOLOv5的models文件夹中）
    :param epochs: 训练轮次（默认100，小数据集可设50-80，大数据集设100-200）
    :param img_size: 输入图像尺寸（与评估一致，固定416）
    :param batch_size: 批次大小（根据GPU显存调整，RTX 4060 Laptop建议16-32）
    :param device: 训练设备（0=GPU，cpu=CPU）
    :param project_dir: 训练结果保存根目录
    :param experiment_name: 实验名称（用于区分不同训练参数）
    """
    # 1. 检查必要文件是否存在
    required_files = [data_yaml_path, model_config_path]
    for file in required_files:
        if not os.path.exists(file):
            print(f"错误：文件不存在 → {file}")
            sys.exit(1)

    # 2. 构建YOLOv5训练命令（调用YOLOv5的train.py）
    # 核心参数说明：
    # --weights: 初始权重（yolov5m.pt=预训练权重，从头训练用--weights ''）
    # --freeze: 冻结前10层（前5轮冻结骨干网络，加快收敛，可选）
    # --save-period: 每5轮保存一次权重，避免意外中断丢失进度
    # 构建YOLOv5训练命令（删除--lr0参数）
    train_command = [
        "python",
        "E:\\DIP_Bee_Detection\\Bee_Detection-main\\Bee_Detection-main\\yolov5-7.0\\yolov5-7.0\\train.py",
        "--data", data_yaml_path,
        "--cfg", model_config_path,
        "--weights", "yolov5m.pt",
        "--epochs", str(epochs),
        "--imgsz", str(img_size),
        "--batch-size", "16",  # 关键：从16降至8，减少单批次内存占用
        "--device", device,
        "--project", project_dir,
        "--name", experiment_name,
        "--save-period", "5",
        "--freeze", "10",
        "--cache", "disk",  # 关键：从ram改为disk，用硬盘缓存数据（而非内存）
        "--optimizer", "SGD",
        "--patience", "10",
        "--workers", "4",  # 关键：从8降至4，减少数据加载进程数，降低内存消耗
        # "--patience", "0",  # 禁用早停，手动控制训练轮数
    ]

    try:
        # 3. 打印训练配置信息
        print("=" * 50)
        print("YOLOv5蜜蜂模型训练配置")
        print("=" * 50)
        print(f"数据集配置：{data_yaml_path}")
        print(f"模型结构：{model_config_path}")
        print(f"训练轮次：{epochs} | 图像尺寸：{img_size} | 批次大小：{batch_size}")
        print(f"训练设备：{device}（GPU：NVIDIA GeForce RTX 4060 Laptop）")
        print(f"结果保存路径：{os.path.join(project_dir, experiment_name)}")
        print("=" * 50)

        # 4. 执行训练命令
        print("开始训练...（首次运行会自动下载预训练权重yolov5m.pt）")
        subprocess.check_call(train_command)

        # 5. 训练完成提示
        result_path = os.path.join(project_dir, experiment_name)
        print(f"\n训练完成！结果保存至：{result_path}")
        print("训练结果包含：")
        print("- weights/：最佳权重（best.pt，用于后续检测/评估）和最后一轮权重（last.pt）")
        print("- results.csv：训练过程指标（loss、mAP、P、R）")
        print("- events.out.tfevents：TensorBoard日志（可可视化训练过程）")
        print("- train_batch0.jpg/val_batch0.jpg：训练/验证批次的可视化结果")

    except subprocess.CalledProcessError as e:
        print(f"\n训练失败：{str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n未知错误：{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # -------------------------- 关键参数配置（根据你的路径修改） --------------------------
    # 1. 数据集配置YAML路径（与评估步骤中的bee_data.yaml一致）
    DATA_YAML = r"E:\DIP_Bee_Detection\archive\train_preprocessed_optimized\bee_data.yaml"

    # 2. YOLOv5模型配置文件路径（使用yolov5m.yaml，中等模型，平衡速度和精度）
    MODEL_CONFIG = r"E:\DIP_Bee_Detection\Bee_Detection-main\Bee_Detection-main\yolov5-7.0\yolov5-7.0\models\yolov5m.yaml"

    # 3. 训练参数（根据你的GPU显存和数据集大小调整）
    EPOCHS = 100  # 蜜蜂数据集若小于1000张图，80轮足够（避免过拟合）
    BATCH_SIZE = 16  # RTX 4060 Laptop 8G显存建议16（若报OOM，改8）
    DEVICE = "0"  # 用GPU训练（必须确保CUDA可用）
    # -----------------------------------------------------------------------------------

    # 启动训练
    train_yolov5_bee_model(
        data_yaml_path=DATA_YAML,
        model_config_path=MODEL_CONFIG,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

