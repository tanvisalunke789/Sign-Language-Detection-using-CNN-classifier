import cv2 
 # OpenCV library for image and video processing
import numpy as np 
 # Numpy library for numerical operations
from keras.models import load_model 
 # Keras library to load the pre-trained model

# Load the trained model
# model = load_model('gesture_recognition_model.h5')
model = load_model('Sign Language ASL Classifier.h5')

# Define the labels for sign language gestures
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Coordinates of the Region of Interest (ROI) in the frame
x, y, w, h = 420, 140, 200, 200

# Function to preprocess the image for prediction
def preprocess_image(img):
    # Use the input image directly (assuming it is in the correct format)
    img_gray = img
    # Resize the image to match the model's expected input size
    img_resized = cv2.resize(img_gray, (224, 224))
    # Normalize the pixel values
    img_normalized = img_resized / 255.0
    # Reshape the image to add batch dimension
    img_reshaped = img_normalized.reshape(1, 224, 224, 3)
    return img_reshaped

# Function to display the predicted label on the image
def display_prediction(img, prediction):
    # Put the predicted label on the frame
    cv2.putText(img, labels[prediction], (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    # Draw a rectangle around the ROI
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # Show the frame with the prediction
    cv2.imshow('Sign Language Recognition', img)

# Function to capture video from webcam and perform prediction
def predict_from_webcam():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        
        if not ret:
            break  # If frame is not captured, break the loop

        # Preprocess the image
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        img_cropped = frame[y:y+h, x:x+w]  # Crop the frame to the ROI
        img_processed = preprocess_image(img_cropped)  # Preprocess the cropped image

        # Perform prediction
        prediction = np.argmax(model.predict(img_processed), axis=-1)
        print(prediction)
        
        # Display the prediction on the frame
        display_prediction(frame, prediction[0])

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()  # Release the webcam
            cv2.destroyAllWindows()  # Close all OpenCV windows
            break

# Run the prediction function
predict_from_webcam()
