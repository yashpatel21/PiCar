import cv2
import numpy as np
import time
import threading
from queue import Queue
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate


# Load the TFLite model and allocate tensors
def load_model(model_path):
    interpreter = Interpreter(
        model_path=model_path, experimental_delegates=[load_delegate("libedgetpu.so.1")]
    )
    interpreter.allocate_tensors()
    return interpreter


# Preprocess the input frame for inference
def preprocess_image(image, target_size):
    resized = cv2.resize(image, target_size)
    return np.expand_dims(resized, axis=0).astype(np.uint8)


# Draw bounding boxes, labels, and FPS on the image
def draw_objects(image, objs, labels, inference_size, fps=None):
    height, width, _ = image.shape

    for obj in objs:
        ymin, xmin, ymax, xmax = obj["bbox"]
        x0, y0, x1, y1 = (
            int(xmin * width),
            int(ymin * height),
            int(xmax * width),
            int(ymax * height),
        )

        # Draw bounding box
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Draw label and confidence
        label = labels.get(obj["class_id"], f"ID {obj['class_id']}")
        confidence = int(obj["score"] * 100)
        text = f"{confidence}% {label}"
        cv2.putText(
            image,
            text,
            (x0, max(0, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    # Draw FPS in the top-left corner
    if fps is not None:
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(
            image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

    return image


# Postprocess outputs to extract bounding boxes, class IDs, and confidence scores
def detect_objects(interpreter, image, threshold):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run inference
    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()

    # Extract outputs
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    class_ids = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]["index"])[0])

    objs = []
    for i in range(num_detections):
        if scores[i] >= threshold:
            objs.append(
                {"bbox": boxes[i], "class_id": int(class_ids[i]), "score": scores[i]}
            )

    return objs


# Capture thread
def capture_thread(cap, frame_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        frame_queue.put(frame)


# Inference thread
def inference_thread(
    interpreter, labels, frame_queue, result_queue, stop_event, input_size
):
    while not stop_event.is_set():
        if frame_queue.empty():
            continue
        frame = frame_queue.get()
        input_data = preprocess_image(frame, input_size)
        objs = detect_objects(interpreter, input_data, threshold=0.5)
        result_queue.put((frame, objs))


# Main function
def main():
    model_path = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
    label_path = "coco_labels.txt"

    # Load labels
    with open(label_path, "r") as f:
        labels = {i: line.strip() for i, line in enumerate(f.readlines())}

    # Load the model
    interpreter = load_model(model_path)
    input_size = interpreter.get_input_details()[0]["shape"][1:3]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to exit.")

    # Queues for threading
    frame_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=1)
    stop_event = threading.Event()

    # Start threads
    capture_thread_handle = threading.Thread(
        target=capture_thread, args=(cap, frame_queue, stop_event)
    )
    inference_thread_handle = threading.Thread(
        target=inference_thread,
        args=(interpreter, labels, frame_queue, result_queue, stop_event, input_size),
    )

    capture_thread_handle.start()
    inference_thread_handle.start()

    # FPS tracking
    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        if not result_queue.empty():
            frame, objs = result_queue.get()
            frame_count += 1
            fps_end_time = time.time()
            elapsed_time = fps_end_time - fps_start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                fps_start_time = fps_end_time
                frame_count = 0

            frame = draw_objects(frame, objs, labels, input_size, fps=fps)
            cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Stop threads
    stop_event.set()
    capture_thread_handle.join()
    inference_thread_handle.join()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
