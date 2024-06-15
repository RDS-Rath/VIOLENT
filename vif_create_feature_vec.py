import os
import numpy as np
import cv2

def vif_create_feature_vec(directory, file_name, sample_rate=5):
    video_path = os.path.join(directory, file_name)
    cap = cv2.VideoCapture(video_path)

    features = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Sample frames based on sample_rate
        if frame_count % sample_rate == 0:
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize frame to reduce dimensionality (example: 100x100)
            resized_frame = cv2.resize(gray_frame, (100, 100))
            # Flatten the frame and add to features
            features.append(resized_frame.flatten())

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    features = np.concatenate(features)
    return features
