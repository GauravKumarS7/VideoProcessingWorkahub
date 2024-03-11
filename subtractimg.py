from skimage.metrics import structural_similarity as compare_ssim
import imutils
import cv2
img1 = cv2.imread("./data/frames0.jpg")
img2 = cv2.imread("./data/frames9.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(gray1, gray2, full=True)
diff = (diff * 255).astype("uint8")
# similarity score
print("SSIM: {}".format(score))
# greater than 0.985 shows barely small changes
thresh = cv2.threshold(diff, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
# show the output images
cv2.imshow("Original", img1)
cv2.imshow("Modified", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
