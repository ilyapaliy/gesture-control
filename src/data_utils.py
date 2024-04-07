import numpy as np

gestures = {1: 'no action', 2: 'cursor', 3: 'left click', 4: 'right click', 5: 'scrolling'}


def preprocess(hand_landmarks):
    def_x, def_y, def_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
    hand_points = np.array([[landmark.x - def_x, landmark.y - def_y, landmark.z - def_z] for landmark in hand_landmarks.landmark])
    return np.asarray(hand_points).reshape(-1)