from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


if __name__ == '__main__':
    model = YOLO('runs/detect/train/weights/best.pt')

    metrics = model.val(
        data='VisDrone.yaml',
        imgsz=640,
        save_json=True,
        save_txt=True,
        max_det=300
    )

    print("\n====== Ultralytics Results ======")
    print(f"mAP50:    {metrics.box.map50 * 100:.1f}")
    print(f"mAP75:    {metrics.box.map75 * 100:.1f}")
    print(f"mAP50-95: {metrics.box.map * 100:.1f}")
