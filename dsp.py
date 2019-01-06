
import numpy as np
import cv2

#import cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')    #codec 
out = cv2.VideoWriter('input.avi',fourcc, 20.0, (640,480))

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #multiScale is used to detect different sizes of images in input.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        
        #w and h represent width and height respectively
        #marking rectangles around the face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        #Region of Image(ROI) of the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Using the eye_Cascades to detect the eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    out.write(img)
            
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        img_name = "output.png".format(0)   
        cv2.imwrite(img_name, img)  #saves the output to the directory
        break

cap.release()
out.release()
cv2.destroyAllWindows()
        

