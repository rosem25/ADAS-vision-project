"""
Generates a synthetic driving video for testing the pipeline.
This simulates a road with lane lines and moving 'car' rectangles.
"""
import cv2
import numpy as np

W, H = 640, 480
FPS = 20
FRAMES = 100

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_drive.mp4', fourcc, FPS, (W, H))

for i in range(FRAMES):
    frame = np.full((H, W, 3), (60, 60, 60), dtype=np.uint8)  # gray road

    # sky
    cv2.rectangle(frame, (0, 0), (W, H // 3), (200, 180, 140), -1)

    # lane lines (converging toward horizon, shifting slightly to simulate motion)
    shift = int(10 * np.sin(i / 10))
    left_bottom = (80 + shift, H)
    left_top = (280, H // 3)
    right_bottom = (W - 80 + shift, H)
    right_top = (360, H // 3)
    cv2.line(frame, left_bottom, left_top, (255, 255, 255), 6)
    cv2.line(frame, right_bottom, right_top, (255, 255, 255), 6)

    # dashed center line
    for y in range(H // 3, H, 40):
        cv2.line(frame, (W // 2, y), (W // 2, y + 20), (0, 255, 255), 4)

    # a "car" moving closer over time (grows in size, moves down/center)
    car_progress = i / FRAMES
    car_w = int(40 + car_progress * 150)
    car_h = int(30 + car_progress * 100)
    car_x = W // 2 - car_w // 2 + int(20 * np.sin(i / 15))
    car_y = int(H // 3 + car_progress * (H * 0.55))
    cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (0, 0, 200), -1)

    out.write(frame)

out.release()
print("test_drive.mp4 created")
