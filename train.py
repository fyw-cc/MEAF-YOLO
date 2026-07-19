from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/MEAF-YOLO/MEAF-YOLO.yaml', task="detect")

    model.train(data='ultralytics/cfg/datasets/VisDrone.yaml',batch=8,epochs=200,imgsz=640)
