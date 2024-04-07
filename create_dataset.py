import cv2
import mediapipe as mp
import numpy as np
import os
from src.data_utils import gestures, preprocess


def save_hand_data(hand_data, gesture_id):
    dataset_path = 'datasets'
    gesture_path = os.path.join(dataset_path, str(gesture_id))
    if not os.path.exists(gesture_path):
        os.makedirs(gesture_path)
    
    files = [f for f in os.listdir(gesture_path) if os.path.isfile(os.path.join(gesture_path, f))]
    file_name = f"{len(files) + 1}.npy"
    file_path = os.path.join(gesture_path, file_name)
    
    # Saving data
    np.save(file_path, np.array(hand_data))
    print(f"Data saved to {file_path}")


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

gesture_id = int(input(f'Gestures: {gestures}\nChoose the gesture number: '))

while True:
    hand_data = []
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_points = [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]
                hand_data = preprocess(hand_landmarks)
                
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(5) & 0xFF == 32:  # Stop the program if you press spacebar
            save_hand_data(hand_data, gesture_id)
            break
    
cap.release()
cv2.destroyAllWindows()
