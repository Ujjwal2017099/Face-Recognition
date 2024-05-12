import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('face-recognition.h5')

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly, ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Resize frame to match the model's expected input
        resized_frame = cv2.resize(frame, (224, 224))  # Change dimensions as required by your model
        resized_frame = resized_frame / 255.0  # Normalize if your model expects normalized inputs
        reshaped_frame = np.expand_dims(resized_frame, axis=0)  # Reshape for model prediction (1, 224, 224, 3)

        # Predict using your model
        predictions = model.predict(reshaped_frame)
        ind = 0  # ind variable to store the index of maximum value in the list
        max_element = predictions[0][0]

        for i in range(1, len(predictions[0])):  # iterate over array
            if predictions[0][i] > max_element:  # to check max value
                max_element = predictions[0][i]
                ind = i
        # Optional: Add prediction or recognition label on frame
        # For example, if predictions is a class label
        print(predictions)
        label = "ujjwal"
        if ind == 1:
            label = "modi"
        if ind == 2:
            label = "rahul"
        cv2.putText(frame, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Video Feed', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
