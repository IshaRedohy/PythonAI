import cv2                                                    # opencv
from random import randrange                                  # for different colors

# Detect all the faces and save their datapoints
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')   

webcam = cv2.VideoCapture(0)                                  #'0' means default webcam! I can also input a video file
while True:                                                   # Iterate forever over frames
    successful_frame_read, frame = webcam.read()              # Read the current frame. first param is a bool,
                                                              # second one is the actual frame
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to gray

    #Detect all the faces in the cascaded file, regardless size and give coordinates of ul and lr
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x,y,w,h) in face_coordinates:
        #(image, ul, lr, color, thickness)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)), 2)  

    cv2.imshow('First face detector', frame)                  # Show the image stored in img
    key = cv2.waitKey(1)                                      # With '1' inside, each frame will be there for
                                                              # for 1 ms
    if key==81 or key==113:
        break                                                 # Break the program when 'q' is pressed

webcam.release()                                              # realese the video captureing object

print("Code compiled sucessfully")