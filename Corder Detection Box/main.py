import numpy as np
import cv2 as cv
import os
import argparse

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
    "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

from nanodet import NanoDet
from deep_sort_realtime.deepsort_tracker import DeepSort

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX, cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN, cv.dnn.DNN_TARGET_NPU]
]

classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


def letterbox(srcimg, target_size=(416, 416)):
    img = srcimg.copy()

    top, left, newh, neww = 0, 0, target_size[0], target_size[1]
    if img.shape[0] != img.shape[1]:
        hw_scale = img.shape[0] / img.shape[1]
        if hw_scale > 1:
            newh, neww = target_size[0], int(target_size[1] / hw_scale)
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            left = int((target_size[1] - neww) * 0.5)
            img = cv.copyMakeBorder(img, 0, 0, left, target_size[1] - neww - left, cv.BORDER_CONSTANT,
                                    value=0)  # add border
        else:
            newh, neww = int(target_size[0] * hw_scale), target_size[1]
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            top = int((target_size[0] - newh) * 0.5)
            img = cv.copyMakeBorder(img, top, target_size[0] - newh - top, 0, 0, cv.BORDER_CONSTANT, value=0)
    else:
        img = cv.resize(img, target_size, interpolation=cv.INTER_AREA)

    letterbox_scale = [top, left, newh, neww]
    return img, letterbox_scale


def unletterbox(bbox, original_image_shape, letterbox_scale):
    ret = bbox.copy()

    h, w = original_image_shape
    top, left, newh, neww = letterbox_scale

    if h == w:
        ratio = h / newh
        ret = ret * ratio
        return ret

    ratioh, ratiow = h / newh, w / neww
    ret[0] = max((ret[0] - left) * ratiow, 0)
    ret[1] = max((ret[1] - top) * ratioh, 0)
    ret[2] = min((ret[2] - left) * ratiow, w)
    ret[3] = min((ret[3] - top) * ratioh, h)

    return ret.astype(np.int32)


def unletterbox_points(keypoint, original_image_shape, letterbox_scale):
    ret = np.array([keypoint.pt[0], keypoint.pt[1]])

    h, w = original_image_shape
    top, left, newh, neww = letterbox_scale

    if h == w:
        ratio = h / newh
        ret = ret * ratio
        return ret

    ratioh, ratiow = h / newh, w / neww
    ret[0] = max((ret[0] - left) * ratiow, 0)
    ret[1] = max((ret[1] - top) * ratioh, 0)

    return ret.astype(np.int32)


def keypoint_extractor(preds, frame, frame_prev, letterbox_scale, fps=None):
    frame_copy = frame.copy()
    frame_copy = cv.cvtColor(frame_copy, cv.COLOR_BGR2RGB)
    keypoints = []
    detector = cv.ORB_create(nfeatures=1000)
    desc_extractor = cv.ORB_create()
    matcher = cv.BFMatcher_create(desc_extractor.defaultNorm())

    for pred in preds:
        bbox = pred[:4]
        conf = pred[-2]
        xmin, ymin, xmax, ymax = unletterbox(bbox, frame_copy.shape[:2], letterbox_scale)
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        # print(f"Bbox coordinates: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
        roi_image = frame_copy[ymin:ymax, xmin:xmax]
        keypoints.append(detector.detect(roi_image))

    return keypoints


def mask_static_keypoints(preds, frame, frame_prev, keypoints, letterbox_scale, th=10):
    ret = frame.copy()
    frame_diff = cv.absdiff(cv.cvtColor(frame, cv.COLOR_BGR2RGB), cv.cvtColor(frame_prev, cv.COLOR_BGR2RGB))
    mask = frame_diff > th
    mask = np.any(mask, axis=-1)
    cv.imshow("Frame Difference", frame_diff)
    cv.imshow("Mask", mask.astype(np.uint8) * 255)

    i = 0
    keypoints_masked = []
    for pred in preds:
        keypoints_masked_bbox = []
        bbox = pred[:4]

        # bbox
        xmin, ymin, xmax, ymax = unletterbox(bbox, ret.shape[:2], letterbox_scale)
        keypoint_bbox = keypoints[i]
        for keypoint in keypoint_bbox:
            x, y = keypoint.pt[0] + xmin, keypoint.pt[1] + ymin

            # keep only keypoints that are in the mask
            if mask[int(y), int(x)]:
                keypoints_masked_bbox.append(keypoint)
        keypoints_masked.append(keypoints_masked_bbox)
        i = i + 1
    return keypoints_masked

def vis(preds, res_img, keypoints, keypoints_unmasked, letterbox_scale, track_ids=None, fps=None):
    ret = res_img.copy()

    # draw FPS
    if fps is not None:
        fps_label = "FPS: %.2f" % fps
        cv.putText(ret, fps_label, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # draw bboxes, labels, IDs and keypoints
    i = 0
    for pred in preds:
        bbox = pred[:4]
        conf = pred[-2]
        classid = pred[-1].astype(np.int32)

        # bbox
        xmin, ymin, xmax, ymax = unletterbox(bbox, ret.shape[:2], letterbox_scale)
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        cv.rectangle(ret, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)

        # label with class and ID
        pid = track_ids[i] if track_ids is not None else -1
        if pid >= 0:
            label = "{:s} ID:{} {:.2f}".format(classes[classid], pid, conf)
        else:
            label = "{:s}: {:.2f}".format(classes[classid], conf)
        cv.putText(ret, label, (xmin, max(ymin - 10, 0)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

        # unmasked (original) keypoints in red
        keypoint_bbox_unmasked = keypoints_unmasked[i]
        for keypoint in keypoint_bbox_unmasked:
            x, y = keypoint.pt[0] + xmin, keypoint.pt[1] + ymin
            cv.circle(ret, (int(x), int(y)), 3, (0, 0, 255), -1)

        # masked keypoints in blue
        keypoint_bbox = keypoints[i]
        for keypoint in keypoint_bbox:
            x, y = keypoint.pt[0] + xmin, keypoint.pt[1] + ymin
            cv.circle(ret, (int(x), int(y)), 3, (255, 0, 0), -1)

        i = i + 1

    return ret

class Track:
    def __init__(self, track_id, bbox):
        self.id = track_id
        # bbox: [xmin, ymin, xmax, ymax] in original frame coordinates
        self.bbox = np.array(bbox, dtype=np.float32)
        self.missed = 0  # how many frames since last update


def iou(b1, b2):
    """
    Intersection over Union between two bboxes [xmin, ymin, xmax, ymax].
    """
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = area1 + area2 - inter + 1e-6
    return inter / union


class SimpleTracker:
    """
    Very basic multi-object tracker using IoU-based data association.
    Keeps an ID per track as long as detections overlap reasonably.
    """

    def __init__(self, iou_threshold=0.3, max_missed=30):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = []
        self.next_id = 0

    def update(self, bboxes):
        """
        bboxes: list of [xmin, ymin, xmax, ymax] for current frame detections.
        Returns a list of track IDs corresponding to each bbox.
        """
        track_ids_for_dets = [-1] * len(bboxes)

        # If no existing tracks, create all new
        if len(self.tracks) == 0:
            for i, bbox in enumerate(bboxes):
                t = Track(self.next_id, bbox)
                self.tracks.append(t)
                track_ids_for_dets[i] = t.id
                self.next_id += 1
            return track_ids_for_dets

        # Greedy matching: each track picks best bbox by IoU
        used_dets = set()
        for track in self.tracks:
            best_det = -1
            best_iou = 0.0
            for d_idx, bbox in enumerate(bboxes):
                if d_idx in used_dets:
                    continue
                score = iou(track.bbox, bbox)
                if score > best_iou:
                    best_iou = score
                    best_det = d_idx

            if best_det != -1 and best_iou > self.iou_threshold:
                # update track
                track.bbox = np.array(bboxes[best_det], dtype=np.float32)
                track.missed = 0
                track_ids_for_dets[best_det] = track.id
                used_dets.add(best_det)
            else:
                track.missed += 1

        # Create new tracks for unmatched detections
        for d_idx, bbox in enumerate(bboxes):
            if d_idx not in used_dets:
                t = Track(self.next_id, bbox)
                self.tracks.append(t)
                track_ids_for_dets[d_idx] = t.id
                self.next_id += 1

        # Remove tracks that disappeared for too long
        self.tracks = [t for t in self.tracks if t.missed < self.max_missed]

        return track_ids_for_dets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Nanodet inference using OpenCV an contribution by Sri Siddarth Chakaravarthy part of GSOC_2022')
    parser.add_argument('--input', '-i', type=str,
                        help='Path to the input image. Omit for using default camera.')
    parser.add_argument('--model', '-m', type=str,
                        default='object_detection_nanodet_2022nov.onnx', help="Path to the model")
    parser.add_argument('--backend_target', '-bt', type=int, default=0,
                        help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
    parser.add_argument('--confidence', default=0.35, type=float,
                        help='Class confidence')
    parser.add_argument('--nms', default=0.6, type=float,
                        help='Enter nms IOU threshold')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Specify to save results. This flag is invalid when using camera.')
    parser.add_argument('--vis', '-v', action='store_true',
                        help='Specify to open a window for result visualization. This flag is invalid when using camera.')
    args = parser.parse_args()

    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    model = NanoDet(modelPath=args.model,
                    prob_threshold=args.confidence,
                    iou_threshold=args.nms,
                    backend_id=backend_id,
                    target_id=target_id)

    tm = cv.TickMeter()
    tm.reset()

    tracker = SimpleTracker(iou_threshold=0.3, max_missed=30)

    # print(args.input)
    if args.input is not None:
        image = cv.imread(args.input)
        input_blob = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Letterbox transformation
        input_blob, letterbox_scale = letterbox(input_blob)

        # Inference
        tm.start()
        preds = model.infer(input_blob)

        tm.stop()
        print("Inference time: {:.2f} ms".format(tm.getTimeMilli()))

        img = vis(preds, image, letterbox_scale)

        if args.save:
            print('Results saved to result.jpg\n')
            cv.imwrite('result.jpg', img)

        if args.vis:
            cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
            cv.imshow(args.input, img)
            cv.waitKey(0)

    else:
        print("Press any key to stop video capture")
        cwd = os.getcwd()
        parent = os.path.abspath(os.path.join(cwd, os.pardir))
        parent_parent = os.path.abspath(os.path.join(parent, os.pardir))
        example_path = f"/Users/calllevels/Desktop/Exchange Courses/TEK5030/Project/examples/example.mp4"
        # print(example_path)
        cap = cv.VideoCapture(example_path)  # (deviceId)

        ##capture camera
        # deviceId = 0
        # cap = cv.VideoCapture(deviceId)
        frame_prev = None

        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break
            if frame_prev is None:
                frame_prev = frame.copy()

            input_blob = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            input_blob, letterbox_scale = letterbox(input_blob)

            # Inference
            tm.start()
            preds = model.infer(input_blob)
            tm.stop()
            fps = tm.getFPS()

            if preds.shape[0] > 0:
                preds = preds[preds[:, -1] == 0]
            else:
                preds = np.zeros((0, 6), dtype=np.float32)

            bboxes = []
            for pred in preds:
                bbox = unletterbox(pred[:4], frame.shape[:2], letterbox_scale)
                xmin, ymin, xmax, ymax = bbox
                bboxes.append([xmin, ymin, xmax, ymax])

            # update tracker and get IDs
            track_ids = tracker.update(bboxes) if len(bboxes) > 0 else []

            # keypoints
            keypoints = keypoint_extractor(preds, frame, frame_prev, letterbox_scale, fps=fps)
            keypoints_masked = mask_static_keypoints(preds, frame, frame_prev, keypoints, letterbox_scale, th=10)

            img = vis(preds, frame, keypoints_masked, keypoints, letterbox_scale, track_ids=track_ids, fps=fps)

            cv.imshow("NanoDet + Simple Tracker", img)

            frame_prev = frame.copy()
            tm.reset()