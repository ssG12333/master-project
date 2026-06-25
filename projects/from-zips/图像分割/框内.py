import cv2
import json

def draw_annotations(image_path, json_path, output_image_path):
    # 加载图片和标注数据
    img = cv2.imread(image_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # 遍历所有标注框并绘制
    for region in annotations:
        xmin, ymin, xmax, ymax = region["bbox"]
        # 绘制矩形框
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # 在框内标注编号
        text_x = xmin + 5  # 在框内左上角
        text_y = ymin + (ymax - ymin) // 2  # 在框内垂直居中
        cv2.putText(img, region["id"], (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 保存结果图像
    cv2.imwrite(output_image_path, img)
    print(f"✅ 标注图像已保存至 {output_image_path}")

# 使用示例
image_path = 'test.png'  # 替换为你的图像路径
json_path = 'regions.json'  # 替换为你的json路径
output_image_path = 'output_with_labels.png'  # 输出图像路径

draw_annotations(image_path, json_path, output_image_path)
