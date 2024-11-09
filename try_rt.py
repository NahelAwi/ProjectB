from source.utils import *

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# angle = 0
# i = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(frame)
        # cv2.imshow("Real-Time Edge Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
