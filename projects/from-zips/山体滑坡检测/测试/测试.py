from ultralytics import YOLO
import cv2
import os


def run_inference():
    # 指定模型和图片路径（请替换为实际路径）
    MODEL_PATH = "best.pt"  # 替换为你的 best.pt 模型路径
    IMAGE_PATH = "test (10).jpg"  # 替换为你的输入图片路径
    CONF_THRESHOLD = 0.3  # 置信度阈值
    OUTPUT_DIR = "output_images"  # 输出文件夹名称

    #类别映射：
    class_names = {
        'Landslides': '山体滑坡'
    }

    # 英文类别名称（用于图片标注）
    class_names_en = list(class_names.keys())

    # 创建输出文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 加载 YOLOv8 模型
    try:
        model = YOLO(MODEL_PATH)
        print("模型加载成功")
    except Exception as e:
        raise ValueError(f"加载模型失败: {str(e)}")

    # 读取图片
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise ValueError(f"无法加载图片: {IMAGE_PATH}")

    # 进行推理
    results = model(image, conf=CONF_THRESHOLD)

    # 处理结果
    for result in results:
        # 检查分类结果
        if result.probs is not None:
            top1_idx = result.probs.top1
            top1_conf = result.probs.top1conf
            predicted_class_en = class_names_en[top1_idx]
            predicted_class_cn = class_names[predicted_class_en]
            print(f"分类结果: {predicted_class_cn} (置信度: {top1_conf:.2f})")

        # 检查检测框
        if result.boxes:
            print(f"检测到 {len(result.boxes)} 个边界框")
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls_idx = int(box.cls[0])
                cls_name_en = class_names_en[cls_idx]
                print(f"边界框: {cls_name_en}, 置信度: {conf:.2f}, 坐标: ({x1}, {y1}, {x2}, {y2})")

                # 绘制矩形框
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制英文标签和置信度
                label = f"{cls_name_en} {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            print("未检测到边界框，显示分类结果")
            # 如果没有检测框，显示英文分类结果在图片顶部
            label = f"{predicted_class_en} {top1_conf:.2f}"
            cv2.putText(image, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 保存输出图片到指定文件夹
    image_name = os.path.basename(IMAGE_PATH)
    output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(image_name)[0]}_output.jpg")
    cv2.imwrite(output_path, image)
    print(f"推理完成！输出图片已保存至: {output_path}")


if __name__ == "__main__":
    run_inference()
