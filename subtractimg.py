import cv2
img1 = cv2.imread("./data/frame0.jpg")
img2 = cv2.imread("./data/frame1.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imshow("black", gray1)
cv2.imshow("color", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
