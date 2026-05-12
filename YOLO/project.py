import numpy as np
import cv2 as cv
import os
import argparse

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
       "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from ultralytics import YOLO








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
    keypoints = mask_static_keypoints(box_corners, frame, frame_prev, keypoints,  th=10)

    ##FLATTENED_KEYPOINTS
    # keypoints_flattend = [kp for sublist in keypoints for kp in sublist]
    # #keypoints_prev_flattend = [kp for sublist in keypoints_prev for kp in sublist]
    # keypoints, frame_descriptors = desc_extractor.compute(frame, keypoints_flattend)
    # if (
    # frame_descriptors is not None and
    # keypoints_descriptors_prev is not None and
    # len(frame_descriptors) > 0 and
    # len(keypoints_descriptors_prev) > 0):
    #     matches = matcher.knnMatch(frame_descriptors, keypoints_descriptors_prev, k=2)
    #     good_matches = extract_good_ratio_matches(matches, max_ratio=0.8)
    # else:
    #     good_matches = []


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

    return keypoints, frame_descriptors, good_matches

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
    

def vis(box_corners, confs,res_img, keypoints, fps=None, keypoints_unmasked=None):
    ret = res_img.copy()

    # draw FPS
    if fps is not None:
        fps_label = "FPS: %.2f" % fps
        cv.putText(ret, fps_label, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # draw bboxes and labels
    i=0
    for (xmin, ymin, xmax, ymax) in box_corners:


        cv.rectangle(ret, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)

        # label
        label = "person {:.2f}".format(confs[i])
        cv.putText(ret, label, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

        # keypoint_bbox_unmasked = keypoints_unmasked[i]
        # for keypoint in keypoint_bbox_unmasked:
        #     x, y = keypoint.pt[0]+xmin, keypoint.pt[1]+ymin
        #     #print(f"Keypoint coordinates: x={x}, y={y}, shape_img={ret.shape}")
        #     cv.circle(ret, (int(x), int(y)), 3, (0, 0, 255), -1)
        

        keypoint_bbox = keypoints[i]
        for keypoint in keypoint_bbox:
            x, y = keypoint.pt[0]+xmin, keypoint.pt[1]+ymin
            #print(f"Keypoint coordinates: x={x}, y={y}, shape_img={ret.shape}")
            cv.circle(ret, (int(x), int(y)), 3, (255, 0, 0), -1)

        i=i+1

    return ret


def vis_matches(box_corner,frame, keypoints, matches, keypoints_prev, letterbox_scale, i, j):
    
    ret = frame.copy()

    #for (xmin, ymin, xmax, ymax) in box_corner:

    
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        (x1, y1) = keypoints[i][img1_idx].pt[0]+xmin, keypoints[i][img1_idx].pt[1]+ymin
        (x2, y2) = keypoints_prev[j][img2_idx].pt[0]+xmin, keypoints_prev[j][img2_idx].pt[1]+ymin

        cv.line(ret, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    return ret




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Nanodet inference using OpenCV an contribution by Sri Siddarth Chakaravarthy part of GSOC_2022')
    parser.add_argument('--input', '-i', type=str,
                        help='Path to the input image. Omit for using default camera.')
    parser.add_argument('--model', '-m', type=str,
                        default='object_detection_nanodet_2022nov.onnx', help="Path to the model")

    parser.add_argument('--confidence', default=0.35, type=float,
                        help='Class confidence')
    parser.add_argument('--nms', default=0.6, type=float,
                        help='Enter nms IOU threshold')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Specify to save results. This flag is invalid when using camera.')
    parser.add_argument('--vis', '-v', action='store_true',
                        help='Specify to open a window for result visualization. This flag is invalid when using camera.')
    args = parser.parse_args()


    model = YOLO("yolo26n.pt")


    tm = cv.TickMeter()
    tm.reset()
    #print(args.input)
    if args.input is not None:
        print("öhm... nnoch nicht implementiert... ")
    else:
        
        print("Press any key to stop video capture")
        cwd = os.getcwd()
        parent = os.path.abspath(os.path.join(cwd, os.pardir))
        parent_parent=os.path.abspath(os.path.join(parent, os.pardir))
        example_path=f"{parent}\examples\example2.mp4"
        #print(example_path)
        cap = cv.VideoCapture(example_path)#(deviceId)

        ##capture camera
        #deviceId = 0
        #cap = cv.VideoCapture(deviceId)
        frame_prev=None
        keypoints_prev=[]
        keypoints_descriptors_prev=None
        #out=cv.VideoWriter('matching.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break
            if frame_prev is None:
                frame_prev=frame.copy()
            
                 

            #input_blob = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            #input_blob, letterbox_scale = letterbox(input_blob)
            
            # Inference
            tm.start()
            preds = model(frame)
            tm.stop()
            #if preds.shape[0] > 0:
            #    preds=preds[preds[:, -1] == 0]
            #print(preds)
            box_corners = []
            confs = []
            for box in preds[0].boxes:
                #detect only people
                cls = int(box.cls[0])
                if cls == 0:
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    box_corners.append((xmin, ymin, xmax, ymax))
                    confs.append(conf)
            keypoints, keypoint_descriptors, good_matches = keypoint_extractor(box_corners, frame, frame_prev, keypoints_prev, keypoints_descriptors_prev,fps=tm.getFPS())
            #keypoints_masked=mask_static_keypoints(preds, frame, frame_prev, keypoints, letterbox_scale, th=10 ) 
           
            

            counts = np.zeros((len(keypoints), len(keypoints_prev)), dtype=int)
            img = vis(box_corners, confs, frame, keypoints, fps=tm.getFPS())

            for i in range(len(keypoints)):
                for j in range(len(keypoints_prev)):
                    #print(f"Good matches between bbox {i} and bbox {j}: {good_matches[i, j]}")
                    if good_matches[i, j] is None:
                        counts[i, j] = 0
                    else:
                        counts[i, j] = len(good_matches[i, j])
            
                    #img = vis_matches(preds,img, keypoints, good_matches[i,j], keypoints_prev, letterbox_scale)
            
            print(f"Counts of good matches between current and previous frame: \n{counts}")
            #TODO Better assignment between boxes (Hungarian algorithm or linear sum assignment) instead of greedy approach
            if counts.size > 0:
                for i in range(len(keypoints)):
                    prev_bbox_index = np.argmax(counts[i])
                    print(f"Best matching previous bbox index for current bbox {i}: {prev_bbox_index} with {counts[i, prev_bbox_index]} good matches")
                    matches_to_prev_bbox = good_matches[i, prev_bbox_index]
                    #img = vis_matches(preds,img, keypoints, matches_to_prev_bbox, keypoints_prev, i, prev_bbox_index)  

            #print(preds)
            cv.imshow("NanoDet Demo", img)
            
            #out.write(img)#save video
            frame_prev=frame.copy()
            keypoints_prev=keypoints
            keypoints_descriptors_prev=keypoint_descriptors

            tm.reset()
