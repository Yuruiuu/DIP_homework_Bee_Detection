# 评估YOLOv5模型在蜜蜂数据集上的性能
# 第一步：先设置环境变量，规避OpenMP冲突（必须放在最开头）
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许重复加载OpenMP库

# 第二步：再导入其他依赖库
import subprocess


def evaluate_bee_model(weights_path, data_yaml_path, img_size=416, device='cpu'):
    """
    评估模型性能（修正数据集读取逻辑，适配蜜蜂数据集）
    :param weights_path: 模型权重路径（必须是.pt文件，如custom_yolov5m.pt）
    :param data_yaml_path: 数据集配置YAML路径（需正确关联图像和标注）
    :param img_size: 输入图像尺寸（与蜜蜂数据集一致，固定416）
    :param device: 运行设备 (cpu或0=GPU，有GPU优先用0加速)
    """
    # 评估结果保存路径（自动创建，避免覆盖）
    project_dir = "bee_evaluation_results"
    name = "val_output"

    # 构建评估命令（补充--task val指定评估测试集，确保读取正确数据）
    command = [
        "python", "yolov5-7.0\\yolov5-7.0\\val.py",  # 修正val.py路径（匹配你本地YOLOv5文件夹结构）
        "--weights", weights_path,  # 正确的模型权重（.pt文件）
        "--data", data_yaml_path,  # 正确的数据集配置YAML
        "--imgsz", str(img_size),  # 与蜜蜂图像尺寸一致（416x416）
        "--device", device,  # 运行设备
        "--conf-thres", "0.001",  # 评估专用低置信度阈值（不漏检）
        "--iou-thres", "0.65",  # IOU阈值（评估标准值）
        "--task", "val",  # 明确指定评估"测试集"（关键参数）
        "--project", project_dir,  # 结果保存根目录
        "--name", name,  # 结果子目录
        "--verbose",  # 可选：打印每类评估结果（便于分析蜜蜂类性能）
        "--save-json"  # 可选：保存COCO格式结果，用于后续对比
    ]

    try:
        # 检查模型权重文件是否存在（提前规避路径错误）
        if not os.path.exists(weights_path) or not weights_path.endswith(".pt"):
            raise FileNotFoundError(f"模型权重路径错误：{weights_path} 不是有效.pt文件")

        # 检查数据集配置文件是否存在
        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"数据集配置文件不存在：{data_yaml_path}")

        # 执行评估命令
        print("开始评估模型性能...")
        print(f"当前评估设备：{device}，图像尺寸：{img_size}")
        subprocess.check_call(command)

        # 评估完成提示
        result_path = os.path.join(project_dir, name)
        print(f"\n评估完成! 结果保存在: {result_path}")
        print("\n关键指标说明（蜜蜂检测核心关注）:")
        print("- mAP@0.5    ：IOU=0.5时的平均精度（蜜蜂框位置准确性，越高越好）")
        print("- mAP@0.5:0.95：IOU=0.5~0.95的平均精度（位置鲁棒性，越高越好）")
        print("- Recall     ：蜜蜂检测召回率（不漏检的能力，越高越好）")

    except Exception as e:
        print(f"\n评估失败: {str(e)}")
        print("请优先检查：1.模型权重路径是否为.pt文件 2.数据集配置YAML是否正确 3.设备是否可用（GPU需驱动正常）")


if __name__ == "__main__":
    # -------------------------- 关键：修改为你的实际路径 --------------------------
    # 1. 模型权重路径：必须是训练好的YOLOv5模型（如custom_yolov5m.pt），不是标注文件夹！
    WEIGHTS_PATH = r"E:\DIP_Bee_Detection\Bee_Detection-main\Bee_Detection-main\models\custom_yolov5m.pt"
    #WEIGHTS_PATH = r"E:\DIP_Bee_Detection\Bee_Detection-main\Bee_Detection-main\bee_training_results\yolov5m_bee_train9\weights\best.pt"


    # 2. 数据集配置YAML路径：需确保YAML内的path、test、labels指向正确
    DATA_YAML_PATH = r"E:\DIP_Bee_Detection\archive\test\bee_data.yaml"
    # -----------------------------------------------------------------------------

    # 执行评估（蜜蜂数据集固定用416尺寸，有GPU则device='0'）
    evaluate_bee_model(
        weights_path=WEIGHTS_PATH,
        data_yaml_path=DATA_YAML_PATH,
        img_size=416,  # 与蜜蜂数据集图像尺寸（416x416）严格一致
        device='0'  # 有NVIDIA GPU填'0'，无GPU填'cpu'
    )


    # WEIGHTS_PATH = r"E:\DIP_Bee_Detection\Bee_Detection-main\Bee_Detection-main\bee_training_results\yolov5m_bee_train18\weights\best.pt"
    # evaluate_bee_model(
    #     weights_path=WEIGHTS_PATH,
    #     data_yaml_path=DATA_YAML_PATH,
    #     img_size=416,  # 与蜜蜂数据集图像尺寸（416x416）严格一致
    #     device='0'  # 有NVIDIA GPU填'0'，无GPU填'cpu'
    # )