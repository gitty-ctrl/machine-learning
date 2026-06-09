import cv2
from cvzone.HandTrackingModule import HandDetector

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

    # Show Camera Feed
    cv2.imshow("Camera Feed", img)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()