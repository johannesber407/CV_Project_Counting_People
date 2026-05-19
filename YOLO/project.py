from unittest import case

import numpy as np
import cv2 as cv
import os
import argparse
from scipy.optimize import linear_sum_assignment
import time
import math

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from ultralytics import YOLO

kalman_filters = []
Tracks = {}

def create_kalman():
    # Initialize Kalman filter parameters
    kalman = cv.KalmanFilter(4, 2)   
 
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Process noise
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5  # Measurement noise

    return kalman

def keypoint_extractor(box_corners, frame, frame_prev,keypoints_prev, keypoints_descriptors_prev, fps=None, th=10):
    frame_copy = frame.copy()
    frame_copy = cv.cvtColor(frame_copy, cv.COLOR_BGR2RGB)
    keypoints = []
    detector = cv.ORB_create(nfeatures=1000)
    desc_extractor = cv.ORB_create()
    matcher = cv.BFMatcher_create(desc_extractor.defaultNorm())


    for (xmin, ymin, xmax, ymax) in box_corners:
        roi_image = frame_copy[ymin:ymax, xmin:xmax]
        keypoints.append(detector.detect(roi_image))



    #mask static keypoints
    keypoints_unmasked = keypoints.copy()
    keypoints = mask_static_keypoints(box_corners, frame, frame_prev, keypoints,  th=20)


    #iterate over combinations of bboxes
    frame_descriptors = []
    for keypoint in keypoints:
        keypoint, frame_descriptors_element = desc_extractor.compute(frame, keypoint)
        frame_descriptors.append(frame_descriptors_element)
    
    good_matches = np.zeros((len(keypoints), len(keypoints_prev)), dtype=object)
    for i in range(len(keypoints)):
        for j in range(len(keypoints_prev)):
            if (
            frame_descriptors[i] is not None and
            keypoints_descriptors_prev[j] is not None):
                
                matches = matcher.knnMatch(frame_descriptors[i], keypoints_descriptors_prev[j], k=2)
                good_matches[i, j] = extract_good_ratio_matches(matches, max_ratio=0.8)
            else:
                good_matches[i, j] = []
                print(f"No descriptors to match for bbox {i} and bbox {j}")

        
    #print(keypoints)

    return keypoints, frame_descriptors, good_matches, keypoints_unmasked

def mask_static_keypoints(box_corners,frame,frame_prev,keypoints,th=10):
    ret=frame.copy()
    frame_diff=cv.absdiff(cv.cvtColor(frame, cv.COLOR_BGR2RGB), cv.cvtColor(frame_prev, cv.COLOR_BGR2RGB))
    mask=frame_diff>th
    mask=np.any(mask, axis=-1)
    cv.imshow("Frame Difference", frame_diff)  
    cv.imshow("Mask", mask.astype(np.uint8)*255)     

    i=0
    keypoints_masked=[]
    for (xmin, ymin, xmax, ymax) in box_corners:
        keypoints_masked_bbox=[]

        keypoint_bbox = keypoints[i]
        for keypoint in keypoint_bbox:
            x, y = keypoint.pt[0]+xmin, keypoint.pt[1]+ymin
            
            #keep only keypoints that are in the mask
            if mask[int(y), int(x)]:
                keypoints_masked_bbox.append(keypoint)
        keypoints_masked.append(keypoints_masked_bbox)
        i=i+1
    return keypoints_masked

def extract_good_ratio_matches(matches, max_ratio, th=20):
    """
    Extracts a set of good matches according to the ratio test.

    :param matches: Input set of matches, the best and the second best match for each putative correspondence.
    :param max_ratio: Maximum acceptable ratio between the best and the next best match.
    :return: The set of matches that pass the ratio test.
    """
    if len(matches) == 0:
        return ()


    matches_arr = np.asarray(matches)
    #distances = np.array([m.distance for m in matches_arr.ravel()]).reshape(matches_arr.shape)
    distances = np.array([[m[0].distance, m[1].distance] for m in matches if len(m) == 2])
    if distances.size == 0:
        return ()
    good_ratio = distances[:, 0] < distances[:, 1] * max_ratio

    good_below_threshold = distances[:, 0] < th

    #print(f"Distances of matches: {distances[:, 0]}")
    good=good_ratio & good_below_threshold

    # Return a tuple of good DMatch objects.
    return tuple(matches_arr[good, 0])
    

def vis(box_corners, confs,res_img, fps=None):
    ret = res_img.copy()

    # draw FPS
    if fps is not None:
        fps_label = "FPS: %.2f" % fps
        cv.putText(ret, fps_label, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    while len(kalman_filters) < len(box_corners):
        kalman_filters.append(create_kalman())
        Tracks[len(kalman_filters) - 1] = (0, 0)

    # draw bboxes and labels
    used_tracks = set()
    i=0
    for (xmin, ymin, xmax, ymax) in box_corners:

        cv.rectangle(ret, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)

        TEST_keypointimg, TEST_Keypoint1, TEST_Descriptors1 = extract_Features(xmin, ymin, xmax, ymax, ret)
        objectCenter = extract_ObjectCenter(xmin, ymin, xmax, ymax, ret, TEST_Keypoint1, TEST_Descriptors1)

        bestDistance = float("inf")
        bestID = 0

        for track_id, (posX, posY) in Tracks.items():
            if track_id in used_tracks:
                continue

            dist = math.sqrt((objectCenter[0] - posX)**2 + (objectCenter[1] - posY)**2)

            if dist < bestDistance:
                bestDistance = dist
                bestID = track_id

        used_tracks.add(bestID)
        Tracks[bestID] = (objectCenter[0], objectCenter[1])
        measured_x, measured_y = objectCenter

        kalman = kalman_filters[bestID]
        kalman.correct(np.array([[np.float32(measured_x)], [np.float32(measured_y)]]))
        predicted = kalman.predict()
        predicted_x, predicted_y = int(predicted[0]), int(predicted[1])
        predicted_dx = float(predicted[2])
        predicted_dy = float(predicted[3])
        predicted_V = math.sqrt(predicted_dx**2 + predicted_dy**2)
            
        label = "person {:.2f}, ID: {}".format(confs[i], bestID)
        cv.putText(ret, label, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

        text = f"Velocity: {predicted_V:.0f}"
        cv.putText(ret, text, (xmin, ymin - 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
        #img=cv.putText(img, text, (10, 125), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if objectCenter:
                
            cv.circle(ret, (measured_x, measured_y), 6, (0, 255, 0), 2)        

        cv.circle(ret, (predicted_x, predicted_y), 8, (255, 0, 0), 2)

        i=i+1

    return ret


def vis_matches(box_corner,frame, keypoints, matches, keypoints_prev,  i, j):
    
    ret = frame.copy()

    xmin, ymin, xmax, ymax=box_corner[i]

    
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        (x1, y1) = keypoints[i][img1_idx].pt[0]+xmin, keypoints[i][img1_idx].pt[1]+ymin
        (x2, y2) = keypoints_prev[j][img2_idx].pt[0]+xmin, keypoints_prev[j][img2_idx].pt[1]+ymin

        cv.line(ret, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    return ret

orb = cv.ORB_create(nfeatures=1000)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
fast = cv.FastFeatureDetector_create(threshold=30)
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

def extract_Features(xmin, ymin, xmax, ymax, res_img):
    roi_image = res_img[ymin +2:ymax-2, xmin+2:xmax-2]
    roi_rgb = cv.cvtColor(roi_image,cv.COLOR_BGR2RGB)

    roi_gray = cv.cvtColor(roi_image,cv.COLOR_BGR2GRAY)

    #keypoints_1 = fast.detect(roi_gray, None)
    # descriptors
    #keypoints_1, descriptors_1 = brief.compute(roi_gray, keypoints_1)

    keypoints_1, descriptors_1 = orb.detectAndCompute(roi_gray, None)

    keypoints_image = cv.drawKeypoints(roi_rgb, keypoints_1, outImage=None, color=(23, 255, 10))

    return keypoints_image, keypoints_1, descriptors_1

def extract_ObjectCenter(xmin, ymin, xmax, ymax, img, keypoint1, descriptor1):
    testimg = img[ymin +2:ymax-2, xmin+2:xmax-2]
    frame_gray = cv.cvtColor(testimg, cv.COLOR_BGR2GRAY)

    keypoints_2, descriptors_2 = orb.detectAndCompute(frame_gray, None)
    #keypoints_2 = fast.detect(frame_gray, None)
    #keypoints_2, descriptors_2 = brief.compute(frame_gray, keypoints_2)

    if descriptors_2 is not None and descriptor1 is not None:
        matches = bf.match(descriptor1, descriptors_2)

        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = matches[:200]

        if good_matches:
            sum_x = 0
            sum_y = 0
            match_count = 0        

            for match in good_matches:
                # .trainIdx gives keypoint index from current frame 
                train_idx = match.trainIdx
                
                # current frame keypoints coordinates
                pt2 = keypoints_2[train_idx].pt
                
                # Sum the x and y coordinates
                sum_x += pt2[0]
                sum_y += pt2[1]
                match_count += 1
            
            # Calculate average of the x and y coordinates
            avg_x = sum_x / match_count + (xmin + 2)
            avg_y = sum_y / match_count + (ymin + 2)

        return int(avg_x),int(avg_y)

    return int(0),int(0)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Nanodet inference using OpenCV an contribution by Sri Siddarth Chakaravarthy part of GSOC_2022')
    parser.add_argument('--input', '-i', type=str,
                        help='Path to the input video. Omit for using default camera.')
    parser.add_argument('--confidence', default=0.35, type=float,
                        help='Class confidence')
    parser.add_argument('--save', '-s', type=str,
                        help='Specify path to save results.')
    parser.add_argument('--orientation', type=str, default='horizontal',
                        help='Choose orientation for counting: "horizontal" or "vertical".')
    args = parser.parse_args()


    model = YOLO("yolo26n.pt")


    tm = cv.TickMeter()
    tm.reset()
    #print(args.input)
    
        
    print("Press any key to stop video capture")
    cwd = os.getcwd()
    # parent = os.path.abspath(os.path.join(cwd, os.pardir))
    # parent_parent=os.path.abspath(os.path.join(parent, os.pardir))
    # example_path=f"{parent}\examples\example2.mp4"
    #print(example_path)
    if args.input is not None:
        example_path=args.input
        cap = cv.VideoCapture(example_path)
    else:
        cap = cv.VideoCapture(0)  # Use default camera
    if args.orientation is not None:
        orientation=args.orientation 
    else:
        orientation="horizontal"

    if args.save is not None:
        out=cv.VideoWriter(args.save, cv.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    confidence_threshold=args.confidence
    frame_prev=None
    keypoints_prev=[]
    keypoints_descriptors_prev=None
    box_centers_prev=None
    count_in=0
    count_out=0

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break
        if frame_prev is None:
            frame_prev=frame.copy()

        # Inference
        tm.start()
        preds = model(frame)
        tm.stop()

        box_corners = []
        box_centers = []
        confs = []
        for box in preds[0].boxes:
            #detect only people
            cls = int(box.cls[0])
            if cls == 0:
                conf = float(box.conf[0])
                if conf < confidence_threshold:
                    continue
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                box_corners.append((xmin, ymin, xmax, ymax))
                box_centers.append(((xmin + xmax) // 2, (ymin + ymax) // 2))
                confs.append(conf)

        img = vis(box_corners, confs, frame, fps=tm.getFPS())
        
        #print(preds)
        label = f"IN: {count_in}  OUT: {count_out}"
        img=cv.putText(img, label, (10, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        cv.imshow("NanoDet Demo", img)
        
        
        if args.save is not None:
            out.write(img)#save video

        frame_prev=frame.copy()
        # keypoints_prev=keypoints
        # keypoints_descriptors_prev=keypoint_descriptors
        box_centers_prev=box_centers

        tm.reset()
