import streamlit as st
import cv2
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
from keras.models import load_model

# Load the trained model
model = load_model('Sign Language ASL Classifier.h5')
#st.write("Model loaded successfully.")

# Define the labels for sign language gestures
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Coordinates of the Region of Interest (ROI)
x, y, w, h = 420, 140, 200, 200

# Function to preprocess the image for prediction
def preprocess_image(img):
    st.write("Preprocessing image...")
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    img_reshaped = img_normalized.reshape(1, 224, 224, 3)
    st.write(f"Image preprocessed: {img_reshaped.shape}")
    return img_reshaped

# Add custom CSS
st.markdown("""
    <style>
    body {
        background-color: #e6f7ff;  /* Light blue background */
    }
    .main {
        background-color: #e6f7ff;
    }
    .sidebar .sidebar-content {
        background-color: #cceeff;  /* Slightly darker blue for sidebar */
    }
    h1 {
        color: #ffffff;
        background-color: #1f77b4;  /* Blue for header */
        padding: 10px;
        text-align: center;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    st.title("Welcome to Sign Language Detection using Gesture Recognition")



    menu = ["Home","Signs Manual",  "Image Uploading", "Webcam Opening"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("""
             
        <div style="background-color:#4CAF50;padding:10px;border-radius:10px">
        <h2 style="color:white;text-align:center;">Home</h2>
        <p style="color:white;text-align:center;">Welcome to the homepage!</p>
        </div>
             
        """, unsafe_allow_html=True)

        st.write("""
             
             
    ***Sign language detection leverages advanced machine learning algorithms to interpret and translate hand gestures into characters. By analyzing images or video streams, the system can recognize and display the meaning of various gestures, bridging the communication gap between users and deaf or mute individuals.***
    
    Here’s a quick overview for the contents of this page:
    
    • **Signs Manual**:
    Access a comprehensive manual of sign language gestures. View a detailed image with various signs to enhance your understanding and learning.
    
    • **Image Uploading**:
    Upload an image of a hand gesture, and our UI will process and predict the sign language character. Simply select an image file, and the system will do the rest!
    
    • **Webcam Opening**:
    Use your webcam to detect sign language gestures in real-time. The application will capture live video, process the gestures, and provide instant predictions on the screen.
    """)

    elif choice == "Image Uploading":
        st.subheader("Upload an Image")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_image is not None:
            width = st.slider("Select the width of the image", 100, 800, 400)
            st.image(uploaded_image, caption="Uploaded Image", width=width)
            # Preprocess the image and make predictions
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            # Perform prediction
            img_processed = preprocess_image(image)
            prediction = model.predict(img_processed)
            prediction_index = np.argmax(prediction, axis=-1)[0]
            predicted_label = labels[prediction_index]
            st.write('PREDICTION', predicted_label)

    elif choice == "Webcam Opening":
        st.subheader("Webcam")

        class VideoProcessor:
            def recv(self, frame):
                try:
                    frame = frame.to_ndarray(format="bgr24")  # Convert frame to numpy array
                    img_cropped = frame[y:y+h, x:x+w]  # Crop the frame to the ROI
                    st.write("Cropped image for prediction.")
                    img_processed = preprocess_image(img_cropped)  # Preprocess the cropped image

                    # Perform prediction
                    prediction = model.predict(img_processed)
                    prediction_index = np.argmax(prediction, axis=-1)[0]
                    predicted_label = labels[prediction_index]
                    st.write(f"Predicted label: {labels[prediction[0]]}")

                    st.markdown(f"**Prediction: {predicted_label} (Index: {prediction_index})**")

                    # Display prediction on the frame
                    cv2.putText(frame, predicted_label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

                except Exception as e:
                    st.write(f"Error in processing frame: {e}")

                return av.VideoFrame.from_ndarray(frame, format='bgr24')  # Return the processed frame

        # WebRTC streamer to capture and display webcam video
        webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
                        rtc_configuration=RTCConfiguration(
                            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                        ))

    elif choice == "Signs Manual":
        st.subheader("Signs Manual")
        st.write("""
        <div style="background-color:#ffcc00;padding:10px;border-radius:10px">
        <h2 style="color:white;text-align:center;">Signs Manual</h2>
        <p style="color:white;text-align:center;">Learn about different sign language gestures here!</p>
        </div>
        """, unsafe_allow_html=True)

        # Display the signs manual image
        manual_image = Image.open("C:\\Users\\tanvi\\Desktop\\sign_manual.jpeg")
        st.image(manual_image, caption="Signs Manual", use_column_width=True)

if __name__ == "__main__":
    main()  # Run the main function
