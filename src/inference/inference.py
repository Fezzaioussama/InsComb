#!/usr/bin/env python3
# real-time water bottle inspection from camera feeds with exported ONNX models
#
# InsComb mode (--part-model): like in the final system of the report, a second model inspects
# the top of the bottle from a second camera, and a bottle is only accepted as clean when both
# models classify it as clean
#
# usage: python -m src.inference.inference models/insbot/inception_v3.onnx --source 0
#        python -m src.inference.inference models/insbot/inception_v3.onnx --source 0 \
#            --part-model models/inspart/inception_v3.onnx --part-source 1
import argparse
import os
import time

import cv2
from PIL import Image

from src.inference.onnx_classifier import OnnxClassifier

os.environ["KMP_DUPLICATE_LIB_OK"]="True"


def open_source(source):
    """
    Open a camera index (e.g. 0), a video file or a GStreamer pipeline (e.g. a CSI camera on the Jetson Nano)
    """
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
    elif '!' in source:
        capture = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    else:
        capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"could not open the video source '{source}'")

    return capture


def classify(classifier, frame):
    """
    Classify an OpenCV (BGR) frame, returns the predicted class name and its probability
    """
    return classifier(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))


def put_text(image, text, y, color):
    """
    Write text on the image at height y, scaled down if it is wider than the image
    """
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
    scale = min(1.0, (image.shape[1] - 20) / width)
    cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def show(frames, results, verdict, clean, fps):
    """
    Display the frames side by side with their predictions and the verdict for the bottle
    """
    height = frames[0].shape[0]
    frames = [cv2.resize(frame, (frame.shape[1] * height // frame.shape[0], height)) for frame in frames]

    for frame, (class_name, probability) in zip(frames, results):
        put_text(frame, f"{class_name} ({probability:.2f})", 30, (255, 255, 255))

    canvas = cv2.hconcat(frames)
    put_text(canvas, f"{verdict}  {fps:.1f} FPS", height - 15, (0, 200, 0) if clean else (0, 0, 255))
    cv2.imshow('Water bottle inspection', canvas)


def main():
    # parse command line
    parser = argparse.ArgumentParser(description='Inspect water bottles from camera feeds in real time')
    parser.add_argument('model', help='path to the ONNX model inspecting the whole bottle (InsBot)')
    parser.add_argument('--source', default='0', help='camera index, video file or GStreamer pipeline (default: 0)')
    parser.add_argument('--labels', default=None, help='class names file of the model (default: labels.txt next to it)')
    parser.add_argument('--part-model', default=None, help='ONNX model inspecting the top of the bottle (InsPart), enables the combined InsComb mode')
    parser.add_argument('--part-source', default='1', help='camera index, video file or GStreamer pipeline of the top-part model (default: 1)')
    parser.add_argument('--part-labels', default=None, help='class names file of the top-part model (default: labels.txt next to it)')
    parser.add_argument('--good-class', default='Propre', help='class name of a bottle without defects (default: Propre)')
    parser.add_argument('--no-display', action='store_true', help='only print the predictions, without opening a window (e.g. over SSH)')
    args = parser.parse_args()

    # one model and one camera per inspected view of the bottle
    classifiers = [OnnxClassifier(args.model, args.labels)]
    sources = [args.source]

    if args.part_model:
        classifiers.append(OnnxClassifier(args.part_model, args.part_labels))
        sources.append(args.part_source)

    for classifier in classifiers:
        print(f"=> {classifier.classes} at {classifier.resolution}x{classifier.resolution} on {classifier.provider}")

        if args.good_class not in classifier.classes:
            parser.error(f"--good-class {args.good_class} is not one of the classes {classifier.classes}")

    try:
        captures = [open_source(source) for source in sources]
    except RuntimeError as error:
        parser.error(str(error))

    fps = 0.0
    last = time.time()

    try:
        while True:
            # capture one frame per view
            frames = []
            for capture in captures:
                ret, frame = capture.read()
                if not ret:
                    break
                frames.append(frame)

            if len(frames) < len(captures):
                print("=> end of the video stream")
                break

            # classify each view, the bottle is clean only when every model says so
            results = [classify(classifier, frame) for classifier, frame in zip(classifiers, frames)]
            defects = [class_name for class_name, _ in results if class_name != args.good_class]
            clean = not defects
            verdict = args.good_class if clean else defects[0]

            # frames per second, smoothed over the last frames
            now = time.time()
            elapsed = max(now - last, 1e-6)
            fps = 0.9 * fps + 0.1 / elapsed if fps else 1 / elapsed
            last = now

            # Print the results to console
            details = '  '.join(f"{class_name} ({probability:.2f})" for class_name, probability in results)
            print(f"bottle={verdict:<10} models: {details}  {fps:.1f} FPS")

            # Display the frames, press q to quit
            if not args.no_display:
                show(frames, results, verdict, clean, fps)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        # When everything done, release the captures
        for capture in captures:
            capture.release()

        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
