import cv2

#cascade file for frontal face recognition
face_cascade = cv2.CascadeClassifier("Faces\haarcascade_frontalface_default.xml")

#turns on webcam
video = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

while True:
    
    #reads the webcam footage
    check, frame = video.read()

    #converts camera footage into grayscale (makes recognition more accurate)
    img_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

    #uses the cascade to check for faces on current frame
    #reutrns an array with x,y cooordinates of the top left most pixel of face 
    #as well the width and height of the face in this order (x , y, width, height)
    faces = face_cascade.detectMultiScale(img_g, scaleFactor = 1.05 , minNeighbors = 5)

    for x,y ,w,h in faces:
        #draws a rectangle around the face 
        #fucntion takes the top left corner and bottom right corner to draw the rectangle 
        #then color using BGR and width of rectangle
        frame = cv2.rectangle(frame,(x,y), (x+w,y+h), (0,0,255), 3)

    #outputs the webcam footage with a rectangle drawn over faces if recognized
    cv2.imshow("capture", frame)
    
    #refreshes frame every 1ms
    key = cv2.waitKey(1)

    #closes loop when q is pressed
    if key == ord('q'):     
        break

#closes camera and webcam window
video.release() 
cv2.destroyAllWindows()