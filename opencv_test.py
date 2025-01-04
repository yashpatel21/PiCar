import cv2
import time

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to exit.")

    # Initialize FPS tracking
    fps_start_time = time.time()
    frame_count = 0
    fps = 0.0  # Initialize fps to avoid unbound variable error

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            break

        # Update FPS counter
        frame_count += 1
        fps_end_time = time.time()
        elapsed_time = fps_end_time - fps_start_time
        if elapsed_time > 1.0:  # Update FPS every second
            fps = frame_count / elapsed_time
            fps_start_time = fps_end_time
            frame_count = 0

        # Draw FPS on the frame
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display the frame
        cv2.imshow("Video Feed", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
