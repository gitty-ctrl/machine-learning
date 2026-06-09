import cv2
from cvzone.HandTrackingModule import HandDetector
import mouse
import numpy as np

# Initialize hand detector
detector = HandDetector(detectionCon=0.9, maxHands=1)

# Initialize webcam
cap = cv2.VideoCapture(0) 
cam_w, cam_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
    if hands:
        # Get the position of the index finger tip
        lmlist = hands[0]['lmList']
        ind_x, ind_y = lmlist[8][0], lmlist[8][1]
        cv2.circle(img, (ind_x,ind_y),5,(255,0,255),2)
        # Move the mouse cursor
        conv_x = int(np.interp(ind_x, [0, cam_w], [0, 1920]))
        conv_y = int(np.interp(ind_y, [0, cam_h], [0, 1080]))
        mouse.move(conv_x, conv_y)

    # Show Camera Feed
    cv2.imshow("Camera Feed", img)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()