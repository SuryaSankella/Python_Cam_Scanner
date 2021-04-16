import cv2
import numpy as np
import mapper
image = cv2.imread("test3.jpeg")  # read in the image
# resizing because opencv does not work well with bigger images
image = cv2.resize(image, (1300, 800))
orig = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB To Gray Scale
cv2.imshow("Title", gray)

# (5,5) is the kernel size and 0 is sigma that determines the amount of blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("Blur", blurred)

# 30 MinThreshold and 50 is the MaxThreshold
edged = cv2.Canny(blurred, 30, 50)
cv2.imshow("Canny", edged)


# retrieve the contours as a list, with simple apprximation model
contours, hierarchy = cv2.findContours(
    edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# the loop extracts the boundary contours of the page
for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*p, True)

    if len(approx) == 4:
        target = approx
        break
approx = mapper.mapp(target)  # find endpoints of the sheet

pts = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]]
                 )  # map to 800*800 target window

# get the top or bird eye view effect
op = cv2.getPerspectiveTransform(approx, pts)
dst = cv2.warpPerspective(orig, op, (800, 800))


cv2.imshow("Scanned", dst)
# press q or Esc to close
cv2.waitKey(0)
cv2.destroyAllWindows()
