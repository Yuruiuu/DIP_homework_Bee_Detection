# 使用YOLOv5模型进行蜜蜂检测
import os
import subprocess


def run_bee_detection(weights_path, data_yaml_path, img_size=416, conf_thres=0.25, device='cpu', image_dir=None):
    """
    运行蜜蜂检测
    :param weights_path: 模型权重路径
    :param data_yaml_path: 数据集配置YAML路径
    :param img_size: 输入图像尺寸
    :param conf_thres: 置信度阈值
    :param device: 运行设备 (cpu或0=GPU)
    :param image_dir: 待检测图像文件夹路径（关键参数）
    """
    # 检测结果保存路径
    project_dir = "bee_detection_results"
    name = "test_output"

    # 构建命令（新增--source参数指定图像路径）
    command = [
        "python", "yolov5-7.0\\yolov5-7.0\\detect.py",
        "--weights", weights_path,
        "--data", data_yaml_path,
        "--source", image_dir,  # 新增：指定待检测图像文件夹
        "--imgsz", str(img_size),
        "--conf-thres", str(conf_thres),
        "--device", device,
        "--save-txt",  # 保存检测结果为txt
        "--save-conf",  # 保存置信度
        "--project", project_dir,
        "--name", name
    ]

    try:
        # 执行检测命令
        print("开始检测蜜蜂图像...")
        subprocess.check_call(command)
        print(f"检测完成! 结果保存在: {os.path.join(project_dir, name)}")
    except Exception as e:
        print(f"检测失败: {str(e)}")


if __name__ == "__main__":
    # 配置路径（请修改为你的实际路径）
    WEIGHTS_PATH = r"E:\DIP_Bee_Detection\Bee_Detection-main\Bee_Detection-main\models\custom_yolov5m.pt"  # 模型权重路径
    DATA_YAML_PATH = r"E:\DIP_Bee_Detection\archive\test\bee_data.yaml"  # 数据集配置文件路径
    IMAGE_DIR = r"E:\DIP_Bee_Detection\archive\test"  # 你的蜜蜂图像文件夹路径（关键修改）

    # 执行检测（传入图像路径参数）
    run_bee_detection(
        weights_path=WEIGHTS_PATH,
        data_yaml_path=DATA_YAML_PATH,
        img_size=416,
        conf_thres=0.25,
        device='0',
        image_dir=IMAGE_DIR  # 传入图像文件夹路径
    )
