# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np
# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9,
}

# load the example image
image = cv2.imread("/home/vince/MachineLearning/opencv_digits/Cam1_test.jpg")
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)

cv2.imshow("Image", image)

# Now split it in two images (env sensor and h2o sensor) (arguments are Y_start:Y_end, X_start:X_end...yeah...)
cropped_env = image[152:221, 79:151]
cropped_h2o = image[237:284, 189:243]

cv2.imshow("cropped_env", cropped_env)
cv2.imshow("cropped_h2o", cropped_h2o)

# apply filters to the images
gray_env = cv2.cvtColor(cropped_env, cv2.COLOR_BGR2GRAY)
blurred_env = cv2.GaussianBlur(gray_env, (5, 5), 0)
edged_env = cv2.Canny(blurred_env, 50, 200, 255)
gray_h2o = cv2.cvtColor(cropped_h2o, cv2.COLOR_BGR2GRAY)
blurred_h2o = cv2.GaussianBlur(gray_h2o, (5, 5), 0)
edged_h2o = cv2.Canny(blurred_h2o, 40, 100, 255) #200 and 50 are thresholds too high for this picture


#cv2.imshow("blurred_env", blurred_env)
#cv2.imshow("edged_env", edged_env)

#cv2.imshow("blurred_h2o", blurred_h2o)
#cv2.imshow("edged_h2o", edged_h2o)

# find contours in the edge map, then sort them by their
# size in descending order
cnts_env = cv2.findContours(edged_env.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts_env = imutils.grab_contours(cnts_env)
cnts_env = sorted(cnts_env, key=cv2.contourArea, reverse=True)
displayCnt_env = None
# loop over the contours
for c in cnts_env:
	# approximate the contour
	peri_env = cv2.arcLength(c, True)
	approx_env = cv2.approxPolyDP(c, 0.02 * peri_env, True)
	# if the contour has four vertices, then we have found
	# the thermostat display
	if len(approx_env) == 4:
		displayCnt_env = approx_env
		break
    
# extract the thermostat display, apply a perspective transform
# to it
warped_env = four_point_transform(gray_env, displayCnt_env.reshape(4, 2))
output_env = four_point_transform(cropped_env, displayCnt_env.reshape(4, 2))


cnts_h2o = cv2.findContours(edged_h2o.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts_h2o = imutils.grab_contours(cnts_h2o)
cnts_h2o = sorted(cnts_h2o, key=cv2.contourArea, reverse=True)
displayCnt_h2o = None
for c in cnts_h2o:
	peri_h2o = cv2.arcLength(c, True)
	approx_h2o = cv2.approxPolyDP(c, 0.02 * peri_h2o, True)
	if len(approx_h2o) == 4:
		displayCnt_h2o = approx_h2o
		break
    
warped_h2o = four_point_transform(gray_h2o, displayCnt_h2o.reshape(4, 2))
output_h2o = four_point_transform(cropped_h2o, displayCnt_h2o.reshape(4, 2))

# Images are resized to get more pixels (better threshold later)
warped_env= imutils.resize(warped_env, height=500)
warped_h2o= imutils.resize(warped_h2o, height=500)

cv2.imshow("warped_env", warped_env)
cv2.imshow("warped_h2o", warped_h2o)
# At this point, the two images are well found
# Now let's find the digits

# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
thresh_env = cv2.adaptiveThreshold(warped_env,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,131,0)
#thresh_env = cv2.threshold(warped_env, 210, 255,
#	cv2.THRESH_BINARY)[1]
kernel_env = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
thresh_env = cv2.morphologyEx(thresh_env, cv2.MORPH_OPEN, kernel_env)

thresh_h2o = cv2.adaptiveThreshold(warped_h2o,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,131,0)
#thresh_h2o = cv2.threshold(warped_h2o, 68, 255,
#	cv2.THRESH_BINARY)[1]

kernel_h2o = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
#thresh_h2o = cv2.erode(thresh_h2o,kernel_h2o,iterations = 1)
thresh_h2o = cv2.morphologyEx(thresh_h2o, cv2.MORPH_OPEN, kernel_h2o)

thresh_env= imutils.resize(thresh_env, height=200)
thresh_h2o= imutils.resize(thresh_h2o, height=200)

thresh_env_humi = thresh_env[41:100, 37:110]
thresh_env_temp = thresh_env[134:174, 60:103]
thresh_h2o_resis = thresh_h2o[53:116, 91:213]

cv2.imshow("thresh_env_humi", thresh_env_humi)
cv2.imshow("thresh_env_temp", thresh_env_temp)
cv2.imshow("thresh_h2o_resis", thresh_h2o_resis)

# find contours in the thresholded image, then initialize the
# digit contours lists
cnts_env = cv2.findContours(thresh_h2o_resis.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts_env = imutils.grab_contours(cnts_env)
digitCnts_env = []
# loop over the digit area candidates
for c in cnts_env:
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)
	print("w:",w,"h:",h)
	# if the contour is sufficiently large, it must be a digit
	if w >= 20 and (h >= 30 and h <= 70):
		digitCnts_env.append(c)

# sort the contours from left-to-right, then initialize the
# actual digits themselves
digitCnts_env = contours.sort_contours(digitCnts_env, method="left-to-right")[0]
digits_env = []

# loop over each of the digits
for c in digitCnts_env:
	# extract the digit ROI
	(x, y, w, h) = cv2.boundingRect(c)
	roi = thresh_h2o_resis[y:y + h, x:x + w]
	# compute the width and height of each of the 7 segments
	# we are going to examine
	(roiH, roiW) = roi.shape
	(dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
	dHC = int(roiH * 0.05)
	# define the set of 7 segments
	segments_env = [
		((0, 0), (w, dH)),	# top
		((0, 0), (dW, h // 2)),	# top-left
		((w - dW, 0), (w, h // 2)),	# top-right
		((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
		((0, h // 2), (dW, h)),	# bottom-left
		((w - dW, h // 2), (w, h)),	# bottom-right
		((0, h - dH), (w, h))	# bottom
	]
	on = [0] * len(segments_env)
	# loop over the segments
	for (i, ((xA, yA), (xB, yB))) in enumerate(segments_env):
		# extract the segment ROI, count the total number of
		# thresholded pixels in the segment, and then compute
		# the area of the segment
		segROI = roi[yA:yB, xA:xB]
		total = cv2.countNonZero(segROI)
		area = (xB - xA) * (yB - yA)
		# if the total number of non-zero pixels is greater than
		# 50% of the area, mark the segment as "on"
		if total / float(area) > 0.5:
			on[i]= 1
			# lookup the digit and draw it on the image
		if tuple(on) in DIGITS_LOOKUP:
			digit = DIGITS_LOOKUP[tuple(on)]
			digits_env.append(digit)
	#cv2.rectangle(thresh_env_humi, (x, y), (x + w, y + h), (0, 255, 0), 1)
	#cv2.putText(thresh_env_humi, str(digit), (x - 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
print(digits_env)

# display the digits
#print(u"{}{}.{} \u00b0C".format(*digits_env))

cv2.waitKey(0)
cv2.destroyAllWindows()
