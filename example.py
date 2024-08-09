from edlib import EDCircles
import cv2

detector = EDCircles("./billiard.jpg")
for e in detector.get_ellipses():
    print(f"center: {e.center}, axes: {e.axes}, theta: {e.theta}")

img = cv2.imread("./billiard.jpg")
for e in detector.get_ellipses():
    int_center = (int(e.center[0]), int(e.center[1]))
    theta_deg = e.theta * 180 / 3.14159265
    cv2.ellipse(img, int_center, e.axes, theta_deg, 0, 360, (0, 255, 0), 2)
cv2.imshow("result", img)
cv2.waitKey(0)
