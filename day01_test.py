from ultralytics import YOLO

# 加载一个轻量级 YOLO 预训练模型
model = YOLO("yolo11n.pt")

# 用一张官方测试图片做目标识别
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True,
    conf=0.25
)

print("识别完成！结果保存在 runs/detect/predict 文件夹里。")