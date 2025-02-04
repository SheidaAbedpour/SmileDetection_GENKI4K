import cv2
import numpy as np
from deepface import DeepFace
import tensorflow as tf

# Load the pre-trained smile detection model
model = tf.keras.models.load_model('smile_non_smile_classifier.h5')

# Initialize the video capture (for a video file)
video_path = "test.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)  # Use 0 for webcam or specify a video file

# Check if the video was opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties (e.g., frame width, height, and FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output video
output_path = "output_video.mp4"  # Replace with your desired output path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Load OpenCV's pre-trained face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face region
        face_roi = frame[y:y + h, x:x + w]

        # Convert the cropped face region to RGB for DeepFace processing
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

        try:
            # Extract face embeddings using DeepFace
            result = DeepFace.represent(img_path=face_rgb, model_name="VGG-Face", enforce_detection=False)

            if result:
                embedding = np.array(result[0]["embedding"]).reshape(1, -1)  # Reshape for model input

                # Predict using the smile detection model
                prediction = model.predict(embedding)
                label = "Smile" if prediction[0] > 0.5 else "No Smile"

                # Draw the label on the frame (on top of the face)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        except Exception as e:
            print(f"Error processing frame: {e}")

    # Write the frame to the output video
    out.write(frame)

    # Display the frame with the bounding box and label
    cv2.imshow("Smile Detection", frame)

    # Press 'q' to quit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer, and close the windows
cap.release()
out.release()
cv2.destroyAllWindows()
