# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
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
	(1, 1, 1, 1, 0, 1, 1): 9
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

cv2.imshow("warped_env", warped_env)
cv2.imshow("warped_h2o", warped_h2o)
# At this point, the two images are well found
# Now let's find the digits

# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
#thresh_env = cv2.adaptiveThreshold(warped_env,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#    cv2.THRESH_BINARY | cv2.THRESH_OTSU,11,0)
thresh_env = cv2.threshold(warped_env, 140, 255,
	cv2.THRESH_BINARY)[1]
kernel_env = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh_env = cv2.morphologyEx(thresh_env, cv2.MORPH_OPEN, kernel_env)

thresh_h2o = cv2.threshold(warped_h2o, 40, 255,
	cv2.THRESH_BINARY)[1]
kernel_h2o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh_h2o = cv2.morphologyEx(thresh_h2o, cv2.MORPH_OPEN, kernel_h2o)

cv2.imshow("thresh_env", thresh_env)
cv2.imshow("thresh_h2o", thresh_h2o)






cv2.waitKey(0)
cv2.destroyAllWindows()
