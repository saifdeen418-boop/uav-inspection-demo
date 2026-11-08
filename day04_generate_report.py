import csv
from datetime import datetime
from pathlib import Path

# 第3天生成的检测统计表
csv_file = Path("detection_summary.csv")

# 第4天要生成的报告文件
txt_report_file = Path("inspection_report.txt")
md_report_file = Path("inspection_report.md")

# 检查统计表是否存在
if not csv_file.exists():
    print("错误：没有找到 detection_summary.csv，请先运行第3天的统计程序。")
    exit()

# 读取CSV统计表
rows = []

with open(csv_file, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# 自动识别有哪些目标类别列
# 排除 image_name 和 summary，剩下的就是 car、person、truck、train 等目标类别
target_columns = [
    col for col in rows[0].keys()
    if col not in ["image_name", "summary"]
]

# 汇总所有图片中的目标数量
total_counts = {}

for col in target_columns:
    total_counts[col] = 0

for row in rows:
    for col in target_columns:
        try:
            total_counts[col] += int(row[col])
        except ValueError:
            total_counts[col] += 0

# 生成报告时间
report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 根据检测结果生成简单结论
total_vehicles = (
    total_counts.get("car", 0)
    + total_counts.get("truck", 0)
    + total_counts.get("bus", 0)
)

total_people = total_counts.get("person", 0)

# 生成文字版报告内容
report_lines = []

report_lines.append("AI无人机巡检识别报告")
report_lines.append("=" * 30)
report_lines.append(f"报告生成时间：{report_time}")
report_lines.append("")
report_lines.append("一、项目说明")
report_lines.append("本报告由AI无人机巡检识别Demo自动生成，基于YOLO模型对航拍图片进行目标检测，并对识别结果进行统计分析。")
report_lines.append("")
report_lines.append("二、检测图片数量")
report_lines.append(f"本次共检测图片数量：{len(rows)} 张")
report_lines.append("")
report_lines.append("三、总体检测结果")

if total_counts:
    for name, count in total_counts.items():
        report_lines.append(f"- {name}：{count} 个")
else:
    report_lines.append("- 未检测到有效目标")

report_lines.append("")
report_lines.append("四、单张图片检测结果")

for row in rows:
    image_name = row["image_name"]
    summary = row["summary"]
    report_lines.append(f"- {image_name}：{summary}")

report_lines.append("")
report_lines.append("五、初步巡检结论")

if total_vehicles > 0:
    report_lines.append(f"本次航拍图片中共识别到车辆相关目标 {total_vehicles} 个，说明图像中存在明显交通或车辆活动。")
else:
    report_lines.append("本次航拍图片中未识别到明显车辆目标。")

if total_people > 0:
    report_lines.append(f"本次航拍图片中共识别到人员目标 {total_people} 个，建议结合巡检场景判断是否存在人员聚集或安全风险。")
else:
    report_lines.append("本次航拍图片中未识别到明显人员目标。")

if total_counts.get("train", 0) > 0 or total_counts.get("truck", 0) > 0:
    report_lines.append("系统识别出 train 或 truck 等目标，考虑到无人机航拍场景存在误检可能，建议进行人工复核。")

report_lines.append("")
report_lines.append("六、模型局限性说明")
report_lines.append("当前使用的是通用轻量YOLO模型，并非专门针对无人机航拍场景训练。")
report_lines.append("在航拍图像中，车辆和人员目标通常较小，且存在视角特殊、目标密集、遮挡和背景复杂等问题，因此可能出现漏检和误检。")
report_lines.append("")
report_lines.append("七、后续优化方向")
report_lines.append("- 更换更大YOLO模型，提高识别精度")
report_lines.append("- 提高输入图片分辨率")
report_lines.append("- 尝试图像切片检测，改善小目标识别")
report_lines.append("- 收集无人机巡检场景数据集")
report_lines.append("- 后续训练或微调专用模型")
report_lines.append("- 将检测结果和报告生成功能整合为完整Demo")

# 写入TXT报告
with open(txt_report_file, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# 写入Markdown报告
md_lines = []

md_lines.append("# AI无人机巡检识别报告")
md_lines.append("")
md_lines.append(f"**报告生成时间：** {report_time}")
md_lines.append("")
md_lines.append("## 一、项目说明")
md_lines.append("本报告由AI无人机巡检识别Demo自动生成，基于YOLO模型对航拍图片进行目标检测，并对识别结果进行统计分析。")
md_lines.append("")
md_lines.append("## 二、检测图片数量")
md_lines.append(f"本次共检测图片数量：**{len(rows)} 张**")
md_lines.append("")
md_lines.append("## 三、总体检测结果")

for name, count in total_counts.items():
    md_lines.append(f"- **{name}**：{count} 个")

md_lines.append("")
md_lines.append("## 四、单张图片检测结果")

for row in rows:
    md_lines.append(f"- `{row['image_name']}`：{row['summary']}")

md_lines.append("")
md_lines.append("## 五、初步巡检结论")

if total_vehicles > 0:
    md_lines.append(f"本次航拍图片中共识别到车辆相关目标 **{total_vehicles} 个**，说明图像中存在明显交通或车辆活动。")
else:
    md_lines.append("本次航拍图片中未识别到明显车辆目标。")

if total_people > 0:
    md_lines.append(f"本次航拍图片中共识别到人员目标 **{total_people} 个**，建议结合巡检场景判断是否存在人员聚集或安全风险。")
else:
    md_lines.append("本次航拍图片中未识别到明显人员目标。")

if total_counts.get("train", 0) > 0 or total_counts.get("truck", 0) > 0:
    md_lines.append("")
    md_lines.append("注意：系统识别出 `train` 或 `truck` 等目标，考虑到无人机航拍场景存在误检可能，建议进行人工复核。")

md_lines.append("")
md_lines.append("## 六、模型局限性说明")
md_lines.append("当前使用的是通用轻量YOLO模型，并非专门针对无人机航拍场景训练。")
md_lines.append("在航拍图像中，车辆和人员目标通常较小，且存在视角特殊、目标密集、遮挡和背景复杂等问题，因此可能出现漏检和误检。")
md_lines.append("")
md_lines.append("## 七、后续优化方向")
md_lines.append("- 更换更大YOLO模型，提高识别精度")
md_lines.append("- 提高输入图片分辨率")
md_lines.append("- 尝试图像切片检测，改善小目标识别")
md_lines.append("- 收集无人机巡检场景数据集")
md_lines.append("- 后续训练或微调专用模型")
md_lines.append("- 将检测结果和报告生成功能整合为完整Demo")

with open(md_report_file, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print("第4天任务完成：巡检报告已自动生成。")
print(f"TXT报告：{txt_report_file}")
print(f"Markdown报告：{md_report_file}")