from typing import List, Tuple


class Ellipse:
    center: Tuple[float, float]
    axes: Tuple[int, int]
    theta: float

class EDCircles:
    def __init__(self, image_path: str): ...
    def get_ellipses(self) -> List[Ellipse]: ...
