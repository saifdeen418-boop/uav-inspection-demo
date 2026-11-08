from ultralytics import YOLO
from pathlib import Path
from collections import Counter
import csv


# =========================
# Day06：本地视频识别 Demo
# 功能：
# 1. 读取 test_videos 文件夹中的视频
# 2. 使用 YOLO 进行视频目标检测
# 3. 保存带识别框的视频结果
# 4. 统计视频中的检测次数
# 5. 生成 video_detection_summary.csv
# =========================


PROJECT_DIR = Path(__file__).resolve().parent

VIDEO_DIR = PROJECT_DIR / "test_videos"
MODEL_PATH = PROJECT_DIR / "yolo11s.pt"

OUTPUT_DIR = PROJECT_DIR / "runs" / "day06_video_detect"
SUMMARY_CSV = PROJECT_DIR / "video_detection_summary.csv"
REPORT_TXT = PROJECT_DIR / "video_inspection_report.txt"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def find_videos():
    """查找 test_videos 文件夹中的视频文件"""
    VIDEO_DIR.mkdir(exist_ok=True)

    videos = [
        file for file in VIDEO_DIR.iterdir()
        if file.suffix.lower() in VIDEO_EXTS
    ]

    return videos


def run_video_detection():
    """运行视频识别并生成统计结果"""

    videos = find_videos()

    if not videos:
        print("没有在 test_videos 文件夹中找到视频。")
        print("请先放入一个 mp4、avi、mov 或 mkv 格式的视频。")
        return

    print("开始加载 YOLO 模型...")
    model = YOLO("yolo11s.pt")
    print("模型加载完成：yolo11s.pt")

    all_rows = []
    all_classes = set()

    print("开始进行本地视频识别...")

    for video_path in videos:
        print(f"\n正在识别视频：{video_path.name}")

        class_counter = Counter()
        frame_count = 0

        results = model.predict(
            source=str(video_path),
            conf=0.15,
            imgsz=1280,
            save=True,
            project=str(OUTPUT_DIR),
            name=video_path.stem,
            exist_ok=True,
            stream=True,
            classes=[2, 5, 7],
        )

        for result in results:
            frame_count += 1

            if result.boxes is None:
                continue

            for cls in result.boxes.cls:
                class_name = result.names[int(cls)]
                class_counter[class_name] += 1
                all_classes.add(class_name)

        total_detection_times = sum(class_counter.values())

        if class_counter:
            summary = "；".join(
                [f"{name}: {count}" for name, count in class_counter.most_common()]
            )
        else:
            summary = "未检测到目标"

        all_rows.append({
            "video_name": video_path.name,
            "frame_count": frame_count,
            "total_detection_times": total_detection_times,
            "summary": summary,
            "class_counter": dict(class_counter)
        })

        print(f"视频帧数统计：{frame_count}")
        print(f"检测次数统计：{summary}")

    write_csv(all_rows, all_classes)
    write_report(all_rows)

    print("\nDay06 本地视频识别完成。")
    print(f"CSV统计表已生成：{SUMMARY_CSV}")
    print(f"文字报告已生成：{REPORT_TXT}")
    print(f"识别结果视频保存在：{OUTPUT_DIR}")


def write_csv(rows, all_classes):
    """生成视频识别统计 CSV"""

    class_columns = sorted(list(all_classes))

    headers = [
        "video_name",
        "frame_count",
        "total_detection_times",
        "summary"
    ] + class_columns

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for row in rows:
            output_row = {
                "video_name": row["video_name"],
                "frame_count": row["frame_count"],
                "total_detection_times": row["total_detection_times"],
                "summary": row["summary"]
            }

            for class_name in class_columns:
                output_row[class_name] = row["class_counter"].get(class_name, 0)

            writer.writerow(output_row)


def write_report(rows):
    """生成简单的视频巡检报告"""

    lines = []

    lines.append("AI无人机巡检视频识别报告（Day06）")
    lines.append("=" * 40)
    lines.append("")
    lines.append("说明：")
    lines.append("本报告统计的是视频逐帧检测次数，不等于真实唯一目标数量。")
    lines.append("例如同一辆车在100帧中都被识别到，可能会被统计为100次。")
    lines.append("当前阶段目标是跑通本地视频识别流程，不代表企业级实时巡检系统。")
    lines.append("")

    for row in rows:
        lines.append(f"视频名称：{row['video_name']}")
        lines.append(f"视频帧数：{row['frame_count']}")
        lines.append(f"总检测次数：{row['total_detection_times']}")
        lines.append(f"检测摘要：{row['summary']}")
        lines.append("-" * 40)

    with open(REPORT_TXT, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


if __name__ == "__main__":
    run_video_detection()