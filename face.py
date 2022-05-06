import face_recognition ## facial recognition lib
import cv2 ## Used to read from webcam
import numpy as np ## jsut numpy things


vid= cv2.VideoCapture(0) # Get webcam
image_lib = [""] # Place your images into the same directory of this script and include the name of the file for instance:"jacky.jpg" into this list
# Create arrays of known face encodings and their names
encoded_face_lib,known_faces,face_locations,face_encodings,face_names= [],[],[],[],[]
# List to store values later
processFrame = True # To process the webcam face
#Loop thru library to recognise images
for image in image_lib:
    loaded_img = face_recognition.load_image_file(image)
    encoded_face_lib.append(face_recognition.face_encodings(loaded_img)[0]) #Train model with my image & Encode the image
    name = image.split('.')
    known_faces.append(name[0]) 
while True: ## Keep running the facial recognition
    ret, frame = vid.read() # Grab image from webcam
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) #Adjsuting frame size, recommended 1/4 for facial recognition procession as per documentations
    rgb_frame = small_frame[:, :, ::-1]#Convert RGB for facial-recogntion package in python
    if processFrame:
        face_locations,face_encoding_list = face_recognition.face_locations(rgb_frame),face_recognition.face_encodings(rgb_frame, face_locations) #Find face location & Encode face in frame
        face_names = [] # Create a list to store all the face names
        for encoding in face_encoding_list:  # Iterate thru face encondings to find perfect match with images in database
            name = "Human Face"
            matches,distance  = face_recognition.compare_faces(encoded_face_lib, encoding),face_recognition.face_distance(encoded_face_lib, encoding) # Check for match for face & Get distance of face from camera
            index_match = np.argmin(distance) #Calculate the nearest distance (Match here is an index like a list)
            if matches[index_match]: # If face matches
                name = known_faces[index_match] #Get the name of the person recognised
            face_names.append(name) #Append these names to face names
    processFrame = not processFrame # Once iteration is done, stop the condition to process the current frame

    # Display the results on screen
    for (top, right, bottom, left), name in zip(face_locations, face_names): ## zip maps face_location to facenames
        top *= 4 ## Scale the frame back to original
        right *= 4 ## Scale the frame back to original
        bottom *= 4  ## Scale the frame back to original
        left *= 4 ## Scale the frame back to original
        ## All these is done to make sure the image returns to normal again
        fontStyle = cv2.FONT_HERSHEY_PLAIN
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0),2) # Create a box around the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,255,0), cv2.FILLED) #Label face with name below
        cv2.putText(frame, name, (left + 6, bottom - 6), fontStyle, 1.0, (0,0,0), 1) # Insert the name on image
    cv2.imshow('Video', frame) # Show results on screen
    if cv2.waitKey(1) & 0xFF == ord('1'): # To end the app, press 1
        break


vid.release() # Webcam now dies
cv2.destroyAllWindows()