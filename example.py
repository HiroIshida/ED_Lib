from edlib import EDCircles
import cv2

detector = EDCircles("./billiard.jpg")
print("detected ellipses:")
for e in detector.get_ellipses():
    print(f"center: {e.center}, axes: {e.axes}, theta: {e.theta}")
print("detected circles:")
for c in detector.get_circles():
    print(f"center: {c.center}, radius: {c.r}")

img = cv2.imread("./billiard.jpg")
for e in detector.get_ellipses():
    int_center = (int(e.center[0]), int(e.center[1]))
    theta_deg = e.theta * 180 / 3.14159265
    cv2.ellipse(img, int_center, e.axes, theta_deg, 0, 360, (0, 255, 0), 2)
for c in detector.get_circles():
    int_center = (int(c.center[0]), int(c.center[1]))
    cv2.circle(img, int_center, int(c.r), (0, 0, 255), 2)
cv2.imshow("result", img)
cv2.waitKey(0)
