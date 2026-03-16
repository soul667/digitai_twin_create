class CameraState:
    pass


class Viewer:
    def __init__(self, *args, **kwargs):
        self.state = type("ViewerState", (), {"status": "running", "num_train_rays_per_sec": 0})()

    def update(self, *args, **kwargs):
        return None
