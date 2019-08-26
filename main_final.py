# -*- coding: utf-8 -*-

# Main.py
import cv2
import DetectChars
import DetectPlates
import pytesseract
import re

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)
showSteps = False


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)


blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()       # attempt KNN training
imgOriginalScene  = cv2.imread("Dataset/19.jpg")        # open image from the dataset
listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates
listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates
cv2.imshow("imgOriginalScene", imgOriginalScene)            # show scene image
if len(listOfPossiblePlates) == 0:                          # if no plates were found
    print("\nno license plates were detected\n")  # inform user no plates were found
else:                                                       # else
    listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)
            # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
    licPlate = listOfPossiblePlates[0]
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    text = pytesseract.image_to_string(licPlate.imgPlate)
    finaldata = re.sub('[":.&%$#@|/!@#$§]', '', text)
#    print(finaldata)
    cv2.imshow("imgPlate", licPlate.imgPlate)           # show crop of plate and threshold of plate
    cv2.imshow("imgThresh", licPlate.imgThresh)
    drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # draw red rectangle around plate
    print("\nlicense plate read from image = " + finaldata + "\n")  # write license plate text to std out
    print("----------------------------------------")
    cv2.imshow("imgOriginalScene", imgOriginalScene)                # re-show scene image
    cv2.imwrite("imgOriginalScene.png", imgOriginalScene)           # write image out to file
# end if else
cv2.waitKey(0)					# hold windows open until user presses a key
cv2.destroyAllWindows()
