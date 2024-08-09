from typing import List, Tuple

class Circle:
    center: Tuple[float, float]
    r: int

class Ellipse:
    center: Tuple[float, float]
    axes: Tuple[int, int]
    theta: float

class EDCircles:
    def __init__(self, image_path: str): ...
    def get_ellipses(self) -> List[Ellipse]: ...
    def get_circles(self) -> List[Circle]: ...
