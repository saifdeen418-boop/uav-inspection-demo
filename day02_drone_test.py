from ultralytics import YOLO

# 加载昨天下载好的YOLO模型
model = YOLO("yolo11n.pt")

# 识别 test_images 文件夹里的所有图片
results = model.predict(
    source="test_images",
    save=True,
    conf=0.15,
    imgsz=1280
)

print("第2天任务完成：无人机场景图片识别已完成。")
print("结果保存在 runs/detect 文件夹下最新的 predict 文件夹中。")