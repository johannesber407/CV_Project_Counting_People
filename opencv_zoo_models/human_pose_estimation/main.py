from ultralytics.utils.plotting import Annotator
import cv2 as cv
import os


def main():
    from ultralytics import YOLO

    model = YOLO('yolov8n-pose.pt')  # load an official model

    # Predict with the model
    cwd = os.getcwd()
    parent = os.path.abspath(os.path.join(cwd, os.pardir))
    parent_parent=os.path.abspath(os.path.join(parent, os.pardir))
    example_path=f"{parent_parent}\examples\example.mp4"
    cap = cv.VideoCapture(example_path)#(deviceId)
    ##capture camera
    #deviceId = 0
    #cap = cv.VideoCapture(deviceId)
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
        results = model(frame)[0]  # predict on an image

        show_img = results.orig_img if len(results) == 0 else None
        for r in results:
            show_img = r.plot(img=show_img)
        cv.imshow('pose', show_img)



if __name__ == '__main__':
    main()