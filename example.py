from edlib import EDCircles

detector = EDCircles("./billiard.jpg")
for e in detector.get_ellipses():
    print(f"center: {e.center}, axes: {e.axes}, theta: {e.theta}")
