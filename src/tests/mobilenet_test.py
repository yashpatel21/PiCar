import numpy as np
import time
import threading
from queue import Queue
import cv2
import gi
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# Init GTK
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf, GLib, Gdk

def load_model(model_path):
    """Initialize TFLite interpreter with Edge TPU acceleration."""
    interpreter = Interpreter(
        model_path=model_path, experimental_delegates=[load_delegate("libedgetpu.so.1")]
    )
    interpreter.allocate_tensors()
    print("Model loaded successfully with Edge TPU delegate")
    return interpreter

def preprocess_image(image, target_size):
    """Resize input frame to target dimensions and prep for inference.
    
    Args:
        image: Input frame in BGR format
        target_size: Model input dimensions (H,W)
    Returns:
        Preprocessed numpy array ready for inference
    """
    resized = cv2.resize(image, target_size)
    return np.expand_dims(resized, axis=0).astype(np.uint8)

def draw_objects(image, objs, labels, inference_size, fps=None):
    """Render detection results on input frame.
    
    Handles BGR->RGB conversion for display compatibility. Input frame 
    arrives in BGR format from PiCamera2 RGB888 configuration.
    
    Args:
        image: Input frame in BGR format
        objs: List of detection results
        labels: Dict mapping class IDs to names 
        inference_size: Model input dimensions
        fps: Optional FPS value to display
    Returns:
        Frame with detection overlay in RGB format
    """
    height, width, _ = image.shape
    draw_image = image.copy()

    # Render detection boxes and labels
    for obj in objs:
        ymin, xmin, ymax, xmax = obj["bbox"]
        x0, y0, x1, y1 = (
            int(xmin * width),
            int(ymin * height),
            int(xmax * width),
            int(ymax * height),
        )

        # BGR color space
        cv2.rectangle(draw_image, (x0, y0), (x1, y1), (0, 255, 0), 2)

        label = labels.get(obj["class_id"], f"ID {obj['class_id']}")
        confidence = int(obj["score"] * 100)
        text = f"{confidence}% {label}"
        cv2.putText(
            draw_image,
            text,
            (x0, max(0, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),  # Red text
            2,
        )

    if fps is not None:
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(
            draw_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

    # Convert to RGB for GTK display
    return cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)

def detect_objects(interpreter, image, threshold):
    """Execute inference and parse detection results.
    
    Args:
        interpreter: TFLite interpreter instance
        image: Preprocessed input tensor
        threshold: Confidence threshold for detections
    Returns:
        List of filtered detection results
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()

    # Parse detection outputs
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    class_ids = interpreter.get_tensor(output_details[1]["index"])[0]
    scores = interpreter.get_tensor(output_details[2]["index"])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]["index"])[0])

    # Filter by confidence threshold
    objs = []
    for i in range(num_detections):
        if scores[i] >= threshold:
            objs.append(
                {"bbox": boxes[i], "class_id": int(class_ids[i]), "score": scores[i]}
            )

    return objs

class ObjectDetectionWindow(Gtk.Window):
    """GTK window implementation for real-time object detection display.
    
    Handles frame rendering using DrawingArea and Pixbuf for optimal performance.
    Expects RGB format input for display compatibility.
    """
    def __init__(self):
        Gtk.Window.__init__(self, title="Object Detection")
        self.set_default_size(640, 480)
        
        self.drawing_area = Gtk.DrawingArea()
        self.add(self.drawing_area)
        self.drawing_area.connect('draw', self.on_draw)
        self.connect("destroy", Gtk.main_quit)
        
        self.current_pixbuf = None
        
    def on_draw(self, widget, cr):
        """Cairo drawing callback for frame rendering."""
        if self.current_pixbuf is not None:
            Gdk.cairo_set_source_pixbuf(cr, self.current_pixbuf, 0, 0)
            cr.paint()
        
    def update_image(self, frame):
        """Update display with new RGB frame data.
        
        Args:
            frame: RGB format numpy array
        """
        height, width, channels = frame.shape
        
        self.current_pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            frame.tobytes(),
            GdkPixbuf.Colorspace.RGB,
            False,
            8,
            width,
            height,
            width * channels,
            None,
            None
        )
        
        self.drawing_area.queue_draw()

def capture_thread(picam, frame_queue, stop_event):
    """Camera capture thread implementation.
    
    Continuously captures frames in BGR format (RGB888 config) and
    maintains single-frame queue buffer for minimal latency.
    """
    while not stop_event.is_set():
        frame = picam.capture_array()
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)

def inference_thread(interpreter, labels, frame_queue, result_queue, stop_event, input_size):
    """Detection inference thread implementation.
    
    Processes frames from capture queue and outputs detection results
    through result queue. Maintains single-item queue for minimal latency.
    """
    while not stop_event.is_set():
        if frame_queue.empty():
            continue
        frame = frame_queue.get()
        input_data = preprocess_image(frame, input_size)
        objs = detect_objects(interpreter, input_data, threshold=0.5)
        if result_queue.full():
            result_queue.get()
        result_queue.put((frame, objs))

def main():
    """Main application entry point and event loop."""
    model_path = "../../models/mobilenet/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
    label_path = "../../models/mobilenet/coco_labels.txt"

    # Load detection class mappings
    with open(label_path, "r") as f:
        labels = {i: line.strip() for i, line in enumerate(f.readlines())}

    # Init model and get input dims
    interpreter = load_model(model_path)
    input_size = interpreter.get_input_details()[0]["shape"][1:3]

    # Configure camera for BGR output via RGB888
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(
        main={"format": 'RGB888', "size": (640, 480)}
    )
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(2)  # Camera init delay

    # Init processing queues
    frame_queue = Queue(maxsize=1)
    result_queue = Queue(maxsize=1)
    stop_event = threading.Event()

    # Create display window
    window = ObjectDetectionWindow()
    window.show_all()

    # FPS calculation vars
    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0

    # Launch processing threads
    capture_thread_handle = threading.Thread(
        target=capture_thread,
        args=(picam2, frame_queue, stop_event)
    )
    inference_thread_handle = threading.Thread(
        target=inference_thread,
        args=(interpreter, labels, frame_queue, result_queue, stop_event, input_size)
    )

    capture_thread_handle.start()
    inference_thread_handle.start()

    def update_display():
        """GTK display update callback."""
        nonlocal frame_count, fps_start_time, fps

        if stop_event.is_set():
            return False

        if not result_queue.empty():
            frame, objs = result_queue.get()
            
            # Update FPS calculation
            frame_count += 1
            fps_end_time = time.time()
            elapsed_time = fps_end_time - fps_start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                fps_start_time = fps_end_time
                frame_count = 0

            # Process and display frame
            frame = draw_objects(frame, objs, labels, input_size, fps=fps)
            
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
                
            window.update_image(frame)

        # Process GTK events
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)

        return True

    # Register display update callback
    GLib.timeout_add(1, update_display)

    try:
        Gtk.main()
    except KeyboardInterrupt:
        print("\nStopping application...")
    finally:
        # Cleanup resources
        stop_event.set()
        capture_thread_handle.join()
        inference_thread_handle.join()
        picam2.stop()
        Gtk.main_quit()

if __name__ == "__main__":
    main()