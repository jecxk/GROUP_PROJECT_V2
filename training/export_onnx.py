from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/smoking_yolo11n/weights/best.pt")

    model.export(
        format="onnx",
        opset=12,
        imgsz=640,
        simplify=True,
        nms=True,      # include NMS trong graph, cho mobile dễ xài
        dynamic=False
    )

if __name__ == "__main__":
    main()
