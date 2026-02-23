__all__ = ["DetectedObject", "LocalObjectDetector"]


def __getattr__(name: str):
    if name in {"DetectedObject", "LocalObjectDetector"}:
        from qimg.detector.object_detector import DetectedObject, LocalObjectDetector

        return {"DetectedObject": DetectedObject, "LocalObjectDetector": LocalObjectDetector}[name]
    raise AttributeError(name)
