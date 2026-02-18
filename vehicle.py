import cv2 as cv
import numpy as np

#webcamera
cap = cv.VideoCapture('video.mp4')



#initialize subtractor
algo = cv.bgsegm.createBackgroundSubtractorMOG() #mog method used for background subtraction


while True:
    ret, frame = cap.read()  #read the video frame by frame 
    #convert to gray scale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3,3), 5) #reduce noise by blurring the image
    #applying on each frame
    img_sub = algo.apply(blur)
    dilated = cv.dilate(img_sub, np.ones((5,5))) #dilation to fill in the gaps
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5)) #structuring element for morphological operations
    dilatada = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel) #closing to remove small holes
    dilatada = cv.morphologyEx(dilatada, cv.MORPH_CLOSE, kernel)
    counterShape = cv.findContours(dilatada, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #find contours in the dilated image










    #cv.imshow('Detector', dilatada) #display the dilated image background opweration result
    #display the output
    cv.imshow('video', frame)

    if cv.waitKey(33) == 27:
        break

cv.destroyAllWindows()
cap.release()
    