import cv2
import numpy as np
import time

def process_frame(frame):

    # for frametime calculation
    start_time = time.time()

    # for performance improvement, resize the frame
    frame = cv2.resize(frame, (320, 240))

    # grayscale conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # edge detection
    edges = cv2.Canny(blurred, 100, 200)

    # define ROI
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width, height // 2),
        (0, height // 2)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, 
    threshold=50, minLineLength=40, maxLineGap=100)
    
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  
            if 0.3 < abs(slope):  # ignore horizontal lines
                filtered_lines.append(line)

    # draw lines
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # combine with original image
    combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    # calculate frametime in ms
    end_time = time.time()
    frame_time = (end_time - start_time) * 1000

    # Display frame time on the frame
    cv2.putText(combined_image, f"{frame_time:.2f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return combined_image

# open camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # process frame
    processed_frame = process_frame(frame)

    # display processed frame
    cv2.imshow("Lane Detection", processed_frame)

    # exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
cap.release()
cv2.destroyAllWindows()
