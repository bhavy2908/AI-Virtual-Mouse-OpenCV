import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # Using webcam no 0

mpHands = mp.solutions.hands
# Creating object called hands
hands = mpHands.Hands()  # Changes can be done in the module hands.py
mpDraw = mp.solutions.drawing_utils  # Drawing 21 point and b/w points draw lines

# FPS
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Sending RGB image to object hand
    results = hands.process(imgRGB)  # Calling the method process in object hands
    print(results.multi_hand_landmarks)  # Just to check the inputs recognized or not

    # Open the object and extract the result within
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:  # handLms --> represents single hand (like a variable)
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape  # Check the dimensions of the image
                cx, cy = int(lm.x*w), int(lm.y*h)  # To print the position in integer pixel value
                print(id, cx, cy)
                # if id == 4:
                #     cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)  # mpHands.HAND_CONNECTIONS --> Connects dots

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
    (255, 0, 255), 3)


    cv2.imshow("Image", img)
    cv2.waitKey(9)  # --> To run the webcam
