import argparse
import cv2
import os
import time

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def main():
    default_model_dir = ''
    default_model = '../../models/mobilenet/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = '../../models/mobilenet/coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use.', default=0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(args.camera_idx)

    # Initialize FPS tracking
    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0  # Initialize fps to avoid unbound variable error

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        # Preprocess and run inference
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]

        # Update FPS counter
        frame_count += 1
        fps_end_time = time.time()
        elapsed_time = fps_end_time - fps_start_time
        if elapsed_time > 1.0:  # Update FPS every second
            fps = frame_count / elapsed_time
            fps_start_time = fps_end_time
            frame_count = 0

        # Annotate frame
        cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels, fps=fps)

        # Display the frame
        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, inference_size, objs, labels, fps=None):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0 + 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    # Draw FPS in the top-left corner
    if fps is not None:
        fps_text = f"FPS: {fps:.2f}"
        cv2_im = cv2.putText(cv2_im, fps_text, (10, 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    return cv2_im

if __name__ == '__main__':
    main()
