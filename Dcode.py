import cv2
import dlib
import numpy as np
from math import hypot
import pygame
import time

cap = cv2.VideoCapture(0)
board = np.zeros((700, 1500), np.uint8)
board[:] = 255

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Keyboard settings
Keyboard = np.zeros((150, 500, 3), np.uint8)
key_set_1 = {
    0: "Q", 1: "W", 2: "E", 3: "R", 4: "T", 5: "Y", 6: "U", 7: "I", 8: "O", 9: "P",
    10: "A", 11: "S", 12: "D", 13: "F", 14: "G", 15: "H", 16: "J", 17: "K", 18: "L", 19: ">",
    20: "Z", 21: "X", 22: "C", 23: "V", 24: "B", 25: "N", 26: "M", 27: ",", 28: ".", 29: "<"
}

def letter(letter_index, text, letter_light):
    # Keys
    x = (letter_index % 10) * 50
    y = (letter_index // 10) * 50
    
    width = 50
    height = 50
    th = 3  # Thickness
    
    if letter_light:
        cv2.rectangle(Keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
    else:
        cv2.rectangle(Keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 0, 0), th)
    
    # Text settingsd
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 3
    font_th = 2
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y
     
    cv2.putText(Keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 0, 0), font_th)

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

font = cv2.FONT_HERSHEY_SIMPLEX

def get_eye_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
        
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
        
    ratio = hor_line_length / ver_line_length
    return ratio

# Counters
frames = 0
letter_index = 0
blinking_frames = 0
text = ""
traverse_speed = 15  # Adjust this value to control traversing speed
blink_frequency = 3  # Lower value increases blink detection frequency

pygame.mixer.init()

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
    Keyboard[:] = (0, 0, 0)
    frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    active_letter = key_set_1[letter_index]    
    
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Eye-blinking
        left_eye_ratio = get_eye_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_eye_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        
        if blinking_ratio > 5.5:
            blinking_frames += 1
            frames -= 1
            
            if blinking_frames == blink_frequency:
                if active_letter == ">":
                    text += " "
                elif active_letter == "<":
                    if len(text) > 0:
                        text = text[:-1]  # Remove the last character
                        text +="/"

                else:
                    text += active_letter
                    pygame.mixer.music.load("sound.wav")
                    pygame.mixer.music.play()
                    time.sleep(1)
            
        else:
            blinking_frames = 0

    # Letters 
    if frames == traverse_speed:
        letter_index += 1
        frames = 0
    if letter_index == 30:
        letter_index = 0
    
    for i in range(30):
        if i == letter_index:
            light = True
        else:
            light = False
        letter(i, key_set_1[i], light)        
             
    cv2.putText(board, text, (10, 100), font, 4, 0, 3)

    cv2.imshow("Frame", frame)
    cv2.imshow("Virtual Keyboard", Keyboard)
    cv2.imshow("Board", board)
 
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release() 
cv2.destroyAllWindows()
