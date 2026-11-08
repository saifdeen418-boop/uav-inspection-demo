from ultralytics import YOLO
from collections import Counter
from pathlib import Path
from datetime import datetime
import csv

# =========================
# 1. 基础路径设置
# =========================

image_folder = Path("test_images")
csv_file = Path("detection_summary.csv")
txt_report_file = Path("inspection_report.txt")
md_report_file = Path("inspection_report.md")

# 检查图片文件夹是否存在
if not image_folder.exists():
    print("错误：没有找到 test_images 文件夹，请先创建并放入测试图片。")
    exit()

# =========================
# 2. 加载YOLO模型并识别图片
# =========================

model = YOLO("yolo11n.pt")

results = model.predict(
    source=str(image_folder),
    save=True,
    conf=0.15,
    imgsz=1280
)

# =========================
# 3. 统计每张图片识别结果
# =========================

all_rows = []
all_classes = set()

print("\n===== 检测结果统计 =====\n")

for result in results:
    image_name = Path(result.path).name
    counts = Counter()

    if result.boxes is not None and len(result.boxes) > 0:
        for cls_id in result.boxes.cls.tolist():
            class_name = result.names[int(cls_id)]
            counts[class_name] += 1
            all_classes.add(class_name)

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

# =========================
# 4. 生成CSV统计表
# =========================

class_list = sorted(list(all_classes))

with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    header = ["image_name"] + class_list + ["summary"]
    writer.writerow(header)

    for row in all_rows:
        line = [row["image_name"]]
        for class_name in class_list:
            line.append(row["counts"].get(class_name, 0))
        line.append(row["summary"])
        writer.writerow(line)

print(f"\n统计表已生成：{csv_file}")

# =========================
# 5. 汇总总数量
# =========================

total_counts = {}

for class_name in class_list:
    total_counts[class_name] = 0

for row in all_rows:
    for class_name in class_list:
        total_counts[class_name] += row["counts"].get(class_name, 0)

total_vehicles = (
    total_counts.get("car", 0)
    + total_counts.get("truck", 0)
    + total_counts.get("bus", 0)
)

total_people = total_counts.get("person", 0)

report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# =========================
# 6. 生成TXT巡检报告
# =========================

report_lines = []

report_lines.append("AI无人机巡检识别报告")
report_lines.append("=" * 30)
report_lines.append(f"报告生成时间：{report_time}")
report_lines.append("")
report_lines.append("一、项目说明")
report_lines.append("本报告由AI无人机巡检识别Demo自动生成，基于YOLO模型对航拍图片进行目标检测，并对识别结果进行统计分析。")
report_lines.append("")
report_lines.append("二、检测图片数量")
report_lines.append(f"本次共检测图片数量：{len(all_rows)} 张")
report_lines.append("")
report_lines.append("三、总体检测结果")

for name, count in total_counts.items():
    report_lines.append(f"- {name}：{count} 个")

report_lines.append("")
report_lines.append("四、单张图片检测结果")

for row in all_rows:
    report_lines.append(f"- {row['image_name']}：{row['summary']}")

report_lines.append("")
report_lines.append("五、初步巡检结论")

if total_vehicles > 0:
    report_lines.append(f"本次航拍图片中共识别到车辆相关目标 {total_vehicles} 个，说明图像中存在明显交通或车辆活动。")
else:
    report_lines.append("本次航拍图片中未识别到明显车辆目标。")

if total_people > 0:
    report_lines.append(f"本次航拍图片中共识别到人员目标 {total_people} 个，建议结合具体巡检场景判断是否存在安全风险。")
else:
    report_lines.append("本次航拍图片中未识别到明显人员目标。")

if total_counts.get("train", 0) > 0 or total_counts.get("truck", 0) > 0:
    report_lines.append("系统识别出 train 或 truck 等目标，考虑到航拍场景存在误检可能，建议人工复核。")

report_lines.append("")
report_lines.append("六、模型局限性说明")
report_lines.append("当前使用的是通用轻量YOLO模型，并非专门针对无人机航拍场景训练。")
report_lines.append("在航拍图像中，车辆和人员目标通常较小，且存在视角特殊、目标密集、遮挡和背景复杂等问题，因此可能出现漏检和误检。")
report_lines.append("")
report_lines.append("七、后续优化方向")
report_lines.append("- 更换更大YOLO模型")
report_lines.append("- 提高输入图片分辨率")
report_lines.append("- 尝试图像切片检测")
report_lines.append("- 收集无人机巡检场景数据集")
report_lines.append("- 后续训练或微调专用模型")
report_lines.append("- 将识别、统计和报告功能整合为完整Demo")

with open(txt_report_file, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# =========================
# 7. 生成Markdown巡检报告
# =========================

md_lines = []

md_lines.append("# AI无人机巡检识别报告")
md_lines.append("")
md_lines.append(f"**报告生成时间：** {report_time}")
md_lines.append("")
md_lines.append("## 一、项目说明")
md_lines.append("本报告由AI无人机巡检识别Demo自动生成，基于YOLO模型对航拍图片进行目标检测，并对识别结果进行统计分析。")
md_lines.append("")
md_lines.append("## 二、检测图片数量")
md_lines.append(f"本次共检测图片数量：**{len(all_rows)} 张**")
md_lines.append("")
md_lines.append("## 三、总体检测结果")

for name, count in total_counts.items():
    md_lines.append(f"- **{name}**：{count} 个")

md_lines.append("")
md_lines.append("## 四、单张图片检测结果")

for row in all_rows:
    md_lines.append(f"- `{row['image_name']}`：{row['summary']}")

md_lines.append("")
md_lines.append("## 五、初步巡检结论")

if total_vehicles > 0:
    md_lines.append(f"本次航拍图片中共识别到车辆相关目标 **{total_vehicles} 个**，说明图像中存在明显交通或车辆活动。")
else:
    md_lines.append("本次航拍图片中未识别到明显车辆目标。")

if total_people > 0:
    md_lines.append(f"本次航拍图片中共识别到人员目标 **{total_people} 个**，建议结合具体巡检场景判断是否存在安全风险。")
else:
    md_lines.append("本次航拍图片中未识别到明显人员目标。")

if total_counts.get("train", 0) > 0 or total_counts.get("truck", 0) > 0:
    md_lines.append("")
    md_lines.append("注意：系统识别出 `train` 或 `truck` 等目标，考虑到无人机航拍场景存在误检可能，建议人工复核。")

md_lines.append("")
md_lines.append("## 六、模型局限性说明")
md_lines.append("当前使用的是通用轻量YOLO模型，并非专门针对无人机航拍场景训练。")
md_lines.append("在航拍图像中，车辆和人员目标通常较小，且存在视角特殊、目标密集、遮挡和背景复杂等问题，因此可能出现漏检和误检。")
md_lines.append("")
md_lines.append("## 七、后续优化方向")
md_lines.append("- 更换更大YOLO模型")
md_lines.append("- 提高输入图片分辨率")
md_lines.append("- 尝试图像切片检测")
md_lines.append("- 收集无人机巡检场景数据集")
md_lines.append("- 后续训练或微调专用模型")
md_lines.append("- 将识别、统计和报告功能整合为完整Demo")

with open(md_report_file, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print("\n第5天补做任务完成：一键流程已跑通。")
print(f"CSV统计表：{csv_file}")
print(f"TXT报告：{txt_report_file}")
print(f"Markdown报告：{md_report_file}")
print("识别结果图片保存在 runs/detect 文件夹下最新的 predict 文件夹中。")