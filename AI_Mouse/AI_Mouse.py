import cv2
from cvzone.HandTrackingModule import HandDetector
import mouse
import numpy as np
import os
import threading
import time

# Initialize hand detector
detector = HandDetector(detectionCon=0.9, maxHands=1)

# Initialize webcam and its dimensions
cap = cv2.VideoCapture(0) 
cam_w, cam_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frameR = 120 # Frame Reduction
l_delay = 0 # Delay for clicking

def l_clk_delay():
    global l_delay
    global l_clk_thread
    time.sleep(1)
    l_delay = 0
    l_clk_thread = threading.Thread(target=l_clk_delay)

l_clk_thread = threading.Thread(target=l_clk_delay)

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (cam_w, cam_h))

# Call the camera
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    out.write(img)

    # Detect hands
    hands, img = detector.findHands(img, flipType=False)
    cv2.rectangle(img, (frameR, frameR), (cam_w - frameR, cam_h - frameR), (255, 0, 255), 2)

    if hands:
        # Get the position of the index finger tip
        lmlist = hands[0]['lmList']
        ind_x, ind_y = lmlist[8][0], lmlist[8][1]
        thm_x, thm_y = lmlist[4][0], lmlist[4][1]
        cv2.circle(img, (ind_x,ind_y),5,(255,0,255),2)
        cv2.circle(img, (thm_x,thm_y),5,(255,0,255),2)
        fingers = detector.fingersUp(hands[0])
        print(fingers)
        print(f"Index Finger: ({ind_x}, {ind_y}), Thumb: ({thm_x}, {thm_y})")
        print(f"Distance: {np.hypot(ind_x - thm_x, ind_y - thm_y)}")
        
        #Move the mouse cursor (Right hand when index is up and thumb is down, flip for LEFT hand)
        if fingers[0] == 0 and fingers[1] == 1:
            conv_x = int(np.interp(ind_x, [frameR, cam_w - frameR], [0, 1920]))
            conv_y = int(np.interp(ind_y, [frameR, cam_h - frameR], [0, 1080]))
            mouse.move(conv_x, conv_y)

        #Click the mouse
        if fingers[0]==1 and fingers[1]==1:
            if abs(ind_x - thm_x) < 25 and abs(ind_y - thm_y) < 25:
                if l_delay ==0:
                    print("Clicking")
                    mouse.click(button='left')
                    l_delay = 1
                    l_clk_thread.start()

    # Show Camera Feed
    cv2.imshow("Camera Feed", img)
    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
os.remove('output.mp4')