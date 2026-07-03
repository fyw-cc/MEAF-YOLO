

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import DetectionModel


class YOLO(Model):
    """YOLO object detection model used by MEAF-YOLO."""

    def __init__(self, model="ultralytics/cfg/MEAF-YOLO/MEAF-YOLO.yaml", task="detect", verbose=False):
        super().__init__(model=model, task=task or "detect", verbose=verbose)

    @property
    def task_map(self):
        """Map the detection task to its model, trainer, validator, and predictor."""
        from ultralytics.models.yolo.detect import DetectionPredictor, DetectionTrainer, DetectionValidator

        return {
            "detect": {
                "model": DetectionModel,
                "trainer": DetectionTrainer,
                "validator": DetectionValidator,
                "predictor": DetectionPredictor,
            },
        }
