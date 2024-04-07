import cv2
import torch
import mediapipe as mp
import numpy as np
import pyautogui
from pynput.mouse import Controller, Button
from src.model import model
from src.data_utils import gestures, preprocess
import time

# Load the trained model
model.load_state_dict(torch.load('gesture_model.pth'))
model.eval()

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Function to get the model prediction
def predict(model, data):
    with torch.no_grad():
        inputs = torch.tensor(data).float().unsqueeze(0)  # Add batch dimension
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

p_x, p_y = 0, 0 # Variables to store previous x, y
ys = []
last_gesture_time = 0
last_gesture = 1
mouse = Controller()


def control_mouse(hand_landmark, gesture_id):
    global p_x, p_y, last_gesture_time, mouse  # Use global variables
    global last_gesture, ys
    screen_width, screen_height = pyautogui.size()  # Get the screen size
    landmark_point = 8
    if gesture_id == 2:
        landmark_point = 4
    x, y = hand_landmark.landmark[landmark_point].x, hand_landmark.landmark[landmark_point].y
    m_x, m_y = mouse.position
    ys.append(y)
    # Check if 2 seconds have passed since the last action for gestures 2 and 3
    if gesture_id in [3, 4] and time.time() - last_gesture_time < 2:
        return  # Skip the action if 3 seconds haven't passed

    x = np.mean([x, p_x])
    y = np.mean([y, p_y])
    x_pos = m_x - ((x - p_x) * screen_width)
    y_pos = m_y + ((y - p_y) * screen_height)

    if gesture_id == 2:
        # Gesture 2: Move the mouse pointer
        x_pos = max(min(screen_width - 10, x_pos), 10)
        y_pos = max(min(screen_height - 10, y_pos), 10)
        mouse.position = (x_pos, y_pos)
    elif gesture_id == 3 and last_gesture == gesture_id:
        # Gesture 3: Left click
        mouse.click(Button.left, 1)
        last_gesture_time = time.time()
    elif gesture_id == 4 and last_gesture == gesture_id:
        # Gesture 4: Right click
        mouse.click(Button.right, 1)
        last_gesture_time = time.time()
    elif gesture_id == 5 and last_gesture == gesture_id:
        # Gesture 5: Scroll up/down
        if p_y is not None:  # Check if the previous y value was set
            if len(ys) < 4:
                return
            if np.mean(ys[-3:]) < ys[-4]:
                mouse.scroll(0, 0.3)  # Scroll up
            elif np.mean(ys[-3:]) > ys[-4]:
                mouse.scroll(0, -0.3)  # Scroll down
    p_x, p_y = x, y  # Update the previous x, y values
    last_gesture = gesture_id

# Capture video from the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the image
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            data = preprocess(hand_landmarks)
            gesture_id = predict(model, data)
            control_mouse(hand_landmarks, gesture_id + 1)
            print(f"Gesture ID: {gestures[gesture_id + 1]}")  # Print the recognized gesture

    # Display the image with hand landmarks
    cv2.imshow('Hand Tracking', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(5) & 0xFF == 27: # Stop program if esc pressed 
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()