import cv2 as cv
from utils import load_and_preprocess_data

# capture = cv.VideoCapture(0)

# while True:
#     isTrue, frame = capture.read()
#     print(frame.shape)
#     cv.imshow("Video", frame)

#     if cv.waitKey(20) & 0xFF == ord("d"):
#         break

# capture.release()
# cv.destroyAllWindows()

load_and_preprocess_data("training_data", "labels.csv", 400, 400)
