import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize Mediapipe hands and drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)

# Variables to keep track of dragging state
dragging = False
drag_start_x, drag_start_y = 0, 0

def get_finger_tip_position(hand_landmarks, index):
    return hand_landmarks.landmark[index].x, hand_landmarks.landmark[index].y

def gesture_to_mouse_action(landmarks, image):
    global dragging, drag_start_x, drag_start_y

    index_finger_tip = get_finger_tip_position(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP)
    thumb_tip = get_finger_tip_position(landmarks, mp_hands.HandLandmark.THUMB_TIP)
    middle_finger_tip = get_finger_tip_position(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP)
    ring_finger_tip = get_finger_tip_position(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP)
    pinky_tip = get_finger_tip_position(landmarks, mp_hands.HandLandmark.PINKY_TIP)
    index_finger_mcp = get_finger_tip_position(landmarks, mp_hands.HandLandmark.INDEX_FINGER_MCP)
    wrist = get_finger_tip_position(landmarks, mp_hands.HandLandmark.WRIST)

    # Convert normalized coordinates to screen coordinates
    x = int(index_finger_tip[0] * screen_width)
    y = int(index_finger_tip[1] * screen_height)
    
    # Move the mouse to the coordinates
    pyautogui.moveTo(x, y, duration=0.1)  # Smooth transition with duration
    
    # Check if the index finger and thumb are close together (pinching gesture) for left click
    if abs(index_finger_tip[0] - thumb_tip[0]) < 0.02 and abs(index_finger_tip[1] - thumb_tip[1]) < 0.02:
        pyautogui.click()
    
    # Check if the middle finger and thumb are close together for right click
    if abs(middle_finger_tip[0] - thumb_tip[0]) < 0.02 and abs(middle_finger_tip[1] - thumb_tip[1]) < 0.02:
        pyautogui.click(button='right')

    # Scroll Up/Down
    if abs(ring_finger_tip[1] - thumb_tip[1]) < 0.02:
        if wrist[1] < ring_finger_tip[1]:
            pyautogui.scroll(-100)  # Scroll down
        else:
            pyautogui.scroll(100)   # Scroll up

    # Drag and Drop
    if abs(index_finger_tip[0] - middle_finger_tip[0]) < 0.02 and abs(index_finger_tip[1] - middle_finger_tip[1]) < 0.02:
        if not dragging:
            dragging = True
            drag_start_x, drag_start_y = x, y
            pyautogui.mouseDown(x, y)
        else:
            pyautogui.moveTo(x, y, duration=0.1)  # Smooth transition with duration
    else:
        if dragging:
            dragging = False
            pyautogui.mouseUp(x, y)

    # Switch Application (Alt + Tab)
    if abs(index_finger_tip[0] - pinky_tip[0]) < 0.02 and abs(index_finger_tip[1] - pinky_tip[1]) < 0.02:
        pyautogui.hotkey('alt', 'tab')

    # Swipe Left/Right (for Navigating Pages or Slides)
    if abs(index_finger_tip[0] - thumb_tip[0]) > 0.2 and abs(index_finger_tip[1] - thumb_tip[1]) < 0.05:
        if index_finger_tip[0] < thumb_tip[0]:
            pyautogui.hotkey('alt', 'left')
        else:
            pyautogui.hotkey('alt', 'right')

    # Swipe Up/Down (for Scrolling)
    if abs(index_finger_tip[1] - thumb_tip[1]) > 0.2 and abs(index_finger_tip[0] - thumb_tip[0]) < 0.05:
        if index_finger_tip[1] < thumb_tip[1]:
            pyautogui.scroll(500)
        else:
            pyautogui.scroll(-500)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB image to get hand landmarks
    result = hands.process(rgb_frame)

    # If hand landmarks are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Perform gesture recognition and map to mouse actions
            gesture_to_mouse_action(hand_landmarks, frame)

    # Display the frame
    cv2.imshow('Hand Gesture Control', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
