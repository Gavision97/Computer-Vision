import os
import cv2
from ultralytics import YOLO

VIDEOS_DIR = os.path.join('data', 'videos')

video_path = os.path.join(VIDEOS_DIR, 'VID_20190902_001755.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape

out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('out', 'detect', 'train', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.2

while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    out.write(frame)
    cv2.imshow('Video', frame)  # Add this line to display the video window
    cv2.waitKey(1)  # Add this line to wait for a short duration

    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
