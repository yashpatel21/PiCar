import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate


# Load the TFLite model and allocate tensors
def load_model(model_path):
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[load_delegate("libedgetpu.so.1")]
    )
    interpreter.allocate_tensors()
    return interpreter


# Preprocess the input frame for inference
def preprocess_image(image, target_size):
    resized = cv2.resize(image, target_size)
    return np.expand_dims(resized, axis=0).astype(np.uint8)


# Draw bounding boxes and labels on the image
def draw_objects(image, objs, labels, inference_size):
    height, width, _ = image.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]

    for obj in objs:
        # Scale bounding box to original image size
        ymin, xmin, ymax, xmax = obj['bbox']
        x0, y0, x1, y1 = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)

        # Draw bounding box
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Draw label and confidence
        label = labels.get(obj['class_id'], f"ID {obj['class_id']}")
        confidence = int(obj['score'] * 100)
        text = f"{confidence}% {label}"
        cv2.putText(image, text, (x0, max(0, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image


# Postprocess outputs to extract bounding boxes, class IDs, and confidence scores
def detect_objects(interpreter, image, threshold):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Extract outputs
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding boxes
    class_ids = interpreter.get_tensor(output_details[1]['index'])[0]  # Class IDs
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])  # Number of detections

    objs = []
    for i in range(num_detections):
        if scores[i] >= threshold:
            objs.append({
                'bbox': boxes[i],
                'class_id': int(class_ids[i]),
                'score': scores[i]
            })

    return objs


# Main function
def main():
    # Model and label paths
    model_path = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
    label_path = "coco_labels.txt"

    # Load labels
    with open(label_path, "r") as f:
        labels = {i: line.strip() for i, line in enumerate(f.readlines())}

    # Load the model
    interpreter = load_model(model_path)
    input_size = interpreter.get_input_details()[0]['shape'][1:3]  # Model input size (e.g., [300, 300])

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to exit.")

    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            break

        # Preprocess frame
        input_data = preprocess_image(frame, input_size)

        # Perform object detection
        objs = detect_objects(interpreter, input_data, threshold=0.5)

        # Draw objects on the frame
        frame = draw_objects(frame, objs, labels, input_size)

        # Display the output
        cv2.imshow("Object Detection", frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
