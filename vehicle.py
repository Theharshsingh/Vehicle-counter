import cv2 as cv
import numpy as np

#webcamera
cap = cv.VideoCapture('video.mp4')

count_line_position = 550 #position of the counting line
min_width_react = 80 #minimum width of the rectangle to be considered a vehicle
min_height_react = 80 #minimum height of the rectangle to be considered a vehicle

#initialize subtractor
algo = cv.bgsegm.createBackgroundSubtractorMOG() #mog method used for background subtraction


def center_handle(x, y, w, h):
    x1 = int(w/2)           #parameters for calculating the center of the rectangle
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []
offset = 6 #allowable error for the counting line
counter = 0

while True:
    ret, frame = cap.read()  #read the video frame by frame 
   
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #convert to gray scale
    blur = cv.GaussianBlur(gray, (3,3), 5) #reduce noise by blurring the image
    #applying on each frame
    img_sub = algo.apply(blur)
    dilated = cv.dilate(img_sub, np.ones((5,5))) #dilation to fill in the gaps
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5)) #structuring element for morphological operations
    dilatada = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel) #closing to remove small holes
    dilatada = cv.morphologyEx(dilatada, cv.MORPH_CLOSE, kernel)
    counterShape,h = cv.findContours(dilatada, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) #find contours in the dilated image




    cv.line(frame, (25, count_line_position), (1200, count_line_position), (255,127,0), 3) #draw the counting line

    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv.boundingRect(c) #get the bounding box of the contour
        validate_counter = (w >= min_width_react) and (h >= min_height_react) #validate the contour based on size

        if not validate_counter:
            continue

        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2) #draw a rectangle around the detected vehicle
        cv.putText(frame, "Vehicle" + str(counter), (x,y-20), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2) #label the detected vehicle

        
        center = center_handle(x, y, w, h) #get the center of the rectangle
        detect.append(center) #add the center to the list of detected centers
        cv.circle(frame, center, 4, (0,0,255), -1) #draw a circle at the center of the rectangle

        for (x, y) in detect:
            if y < (count_line_position + offset) and y > (count_line_position - offset): #check if the center is near the counting line
                counter += 1 #increment the counter
                cv.line(frame, (25, count_line_position), (1200, count_line_position), (0,127,255), 3) #change the color of the counting line to indicate a count
                detect.remove((x,y)) #remove the center from the list of detected centers
                print("Vehicle Count: " + str(counter)) #print the current count of vehicles

    cv.putText(frame, "VEHICLE COUNT: " + str(counter), (450,70), cv.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5) #display the vehicle count on the frame

    #cv.imshow('Detector', dilatada) #display the dilated image background opweration result
    #display the output
    cv.imshow('video', frame)

    if cv.waitKey(33) == 27:
        break

cv.destroyAllWindows()
cap.release()
    