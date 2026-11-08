from ultralytics import YOLO
from collections import Counter
from pathlib import Path
import csv

# 加载YOLO模型
model = YOLO("yolo11n.pt")

# 识别 test_images 文件夹里的图片
results = model.predict(
    source="test_images",
    save=True,
    conf=0.15,
    imgsz=1280
)

# 用来保存每张图片的统计结果
all_rows = []
all_classes = set()

print("\n===== 检测结果统计 =====\n")

for result in results:
    image_name = Path(result.path).name
    counts = Counter()

    # 如果识别到了目标
    if result.boxes is not None and len(result.boxes) > 0:
        for cls_id in result.boxes.cls.tolist():
            class_name = result.names[int(cls_id)]
            counts[class_name] += 1
            all_classes.add(class_name)

    # 生成终端显示内容
    if counts:
        summary = "，".join([f"{name}: {num}" for name, num in counts.items()])
    else:
        summary = "未识别出目标"

    print(f"{image_name}：{summary}")

    all_rows.append({
        "image_name": image_name,
        "counts": dict(counts),
        "summary": summary
    })

# 生成CSV表格
class_list = sorted(list(all_classes))
csv_file = "detection_summary.csv"

with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    # 表头
    header = ["image_name"] + class_list + ["summary"]
    writer.writerow(header)

    # 每张图的结果
    for row in all_rows:
        line = [row["image_name"]]
        for class_name in class_list:
            line.append(row["counts"].get(class_name, 0))
        line.append(row["summary"])
        writer.writerow(line)

print(f"\n统计完成！结果已保存到：{csv_file}")
print("识别后的图片保存在 runs/detect 文件夹下最新的 predict 文件夹中。")