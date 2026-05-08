import numpy as np
import cv2 as cv
import os
import argparse

# Your NanoDet wrapper (must be available as a module)
from nanodet import NanoDet

# Deep SORT realtime tracker
from deep_sort_realtime.deepsort_tracker import DeepSort

# Check OpenCV version
opencv_python_version = lambda str_version: tuple(map(int, (str_version.split("."))))
assert opencv_python_version(cv.__version__) >= opencv_python_version("4.10.0"), \
    "Please install latest opencv-python for benchmark: python3 -m pip install --upgrade opencv-python"

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
            img = cv.copyMakeBorder(img, 0, 0, left, target_size[1] - neww - left,
                                    cv.BORDER_CONSTANT, value=0)
        else:
            newh, neww = int(target_size[0] * hw_scale), target_size[1]
            img = cv.resize(img, (neww, newh), interpolation=cv.INTER_AREA)
            top = int((target_size[0] - newh) * 0.5)
            img = cv.copyMakeBorder(img, top, target_size[0] - newh - top, 0, 0,
                                    cv.BORDER_CONSTANT, value=0)
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
        return ret.astype(np.int32)

    ratioh, ratiow = h / newh, w / neww
    ret[0] = max((ret[0] - left) * ratiow, 0)
    ret[1] = max((ret[1] - top) * ratioh, 0)
    ret[2] = min((ret[2] - left) * ratiow, w)
    ret[3] = min((ret[3] - top) * ratioh, h)

    return ret.astype(np.int32)


def vis_tracks(frame, tracks, fps=None):
    """
    Draw Deep SORT tracks on the frame.
    'tracks' is a list of Track objects from deep-sort-realtime.
    """
    ret = frame.copy()

    # draw FPS
    if fps is not None:
        fps_label = "FPS: %.2f" % fps
        cv.putText(ret, fps_label, (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for t in tracks:
        if not t.is_confirmed():
            continue
        # ltrb: [left, top, right, bottom]
        ltrb = t.to_ltrb()
        xmin, ymin, xmax, ymax = map(int, ltrb)

        cv.rectangle(ret, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        track_id = t.track_id
        label = f"Person ID:{track_id}"
        cv.putText(ret, label, (xmin, max(ymin - 10, 0)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return ret


def main():
    parser = argparse.ArgumentParser(
        description='NanoDet + Deep SORT multi-person tracking demo '
                    '(check if it is the same person walking in/out).')
    parser.add_argument('--input', '-i', type=str,
                        help='Path to the input image or video. '
                             'If omitted, a default video path is used.')
    parser.add_argument('--model', '-m', type=str,
                        default='object_detection_nanodet_2022nov.onnx',
                        help="Path to the NanoDet ONNX model")
    parser.add_argument('--backend_target', '-bt', type=int, default=0,
                        help='''Choose one of the backend-target pair:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
    parser.add_argument('--confidence', default=0.35, type=float,
                        help='Class confidence threshold for NanoDet')
    parser.add_argument('--nms', default=0.6, type=float,
                        help='NMS IOU threshold for NanoDet')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Save result image or video.')
    parser.add_argument('--vis', '-v', action='store_true',
                        help='Open a window for result visualization.')

    args = parser.parse_args()

    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # Init NanoDet
    model = NanoDet(modelPath=args.model,
                    prob_threshold=args.confidence,
                    iou_threshold=args.nms,
                    backend_id=backend_id,
                    target_id=target_id)

    # Init Deep SORT tracker
    deepsort = DeepSort(
        max_age=30,
        n_init=3,
        max_cosine_distance=0.2,
        nn_budget=100,
        override_track_class=None
    )

    tm = cv.TickMeter()
    tm.reset()
    # VIDEO MODE
    if args.input is None:
        # Your default video path; change as needed.
        example_path = "/Users/calllevels/Desktop/Exchange Courses/TEK5030/Project/examples/example.mp4"
        cap = cv.VideoCapture(example_path)
    else:
        cap = cv.VideoCapture(args.input)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Setup video writer if saving
    writer = None
    if args.save:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        fps_out = cap.get(cv.CAP_PROP_FPS)
        if fps_out <= 0:
            fps_out = 25
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        writer = cv.VideoWriter('result_video.mp4', fourcc, fps_out, (width, height))
        print("Saving result to result_video.mp4")

    print("Press ESC (in the window) to stop.")
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        input_blob = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        input_blob, letterbox_scale = letterbox(input_blob)

        tm.start()
        preds = model.infer(input_blob)
        tm.stop()
        fps = tm.getFPS()
        tm.reset()

        # Filter to persons only
        if preds.shape[0] > 0:
            preds = preds[preds[:, -1] == 0]
        else:
            preds = np.zeros((0, 6), dtype=np.float32)

        detections = []
        for pred in preds:
            bbox = unletterbox(pred[:4], frame.shape[:2], letterbox_scale)
            xmin, ymin, xmax, ymax = bbox
            w = xmax - xmin
            h = ymax - ymin
            conf = float(pred[-2])
            if conf < args.confidence:
                continue
            x_c = xmin + w / 2.0
            y_c = ymin + h / 2.0
            detections.append(([x_c, y_c, w, h], conf, 'person'))

        tracks = deepsort.update_tracks(detections, frame=frame)
        img_vis = vis_tracks(frame, tracks, fps=fps)

        if args.vis:
            cv.imshow("NanoDet + Deep SORT", img_vis)
            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        if writer is not None:
            writer.write(img_vis)

    cap.release()
    if writer is not None:
        writer.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
