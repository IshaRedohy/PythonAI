import cv2                                              #opencv
from random import randrange                            #for different colors

#Detect all the faces and save their datapoints
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')   

img = cv2.imread('mult.webp')                           #Read the image and store it in variable img
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #Convert to gray

#Detect all the faces in the cascaded file, regardless size and give coordinates of ul and lr
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
#print(face_coordinates)                                #First 2 upper left, 2 lower right

# (x,y,w,h) = face_coordinates[0]                       #storing the face coordinates.(it's an ARRAY OF FACES DETECTED)
for (x,y,w,h) in face_coordinates:
    #(image, ul, lr, color, thickness)
    cv2.rectangle(img, (x,y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)), 2)  

cv2.imshow('First face detector', img)                  #Show the image stored in img
cv2.waitKey()                                           #Program will wait till a key is pressed before it 
                                                        #closes the image in a blink

print("Code compiled sucessfully")