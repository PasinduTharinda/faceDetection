import cv2
from deepface import DeepFace

# Load the pre-trained emotion analysis model
model = DeepFace.build_model("Emotion")

# Initialize the webcam
video = cv2.VideoCapture(0)

# Create a list of emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

while video.isOpened():
    _, frame = video.read()

    # Detect faces in the frame
    face_cascade = cv2.CascadeClassifier("Model/haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        face_roi = frame[y:y + h, x:x + w]
        # Resize the face to the required input size for the model
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = face_roi.reshape(1, 48, 48, 1)

        # Make an emotion prediction
        emotion_predictions = model.predict(face_roi)

        # Get the predicted emotion label
        predicted_emotion = emotion_labels[emotion_predictions.argmax()]

        # Display the predicted emotion label on the frame
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv2.imshow('Emotion Detection', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()



# # imgpath='1.jpg'
# # image=cv2.imread(imgpath)

# # # check imotin presentages
# # analyze = DeepFace.analyze(image,actions=['emotion'])
# # # print(analyze)
# # # print(type(analyze))
# # # print(analyze['dominant_emotion'])
# # for item in analyze:
# #     if 'dominant_emotion' in item:
# #         print(item['dominant_emotion'])
