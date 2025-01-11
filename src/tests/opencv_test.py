from picamera2 import Picamera2
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize the Picamera2 object
picam2 = Picamera2()
picam2.start()

# Set up Matplotlib figure
fig, ax = plt.subplots()
ax.axis("off")  # Hide axis for cleaner display
img_display = ax.imshow([[0]])  # Initialize with an empty image

# Flag to control the animation loop
stop_animation = False

# Function to handle key press events
def on_key(event):
    global stop_animation
    if event.key == "q":  # Press 'q' to quit
        stop_animation = True
        plt.close()

# Connect the key press event to the handler
fig.canvas.mpl_connect("key_press_event", on_key)

# Update function for Matplotlib animation
def update(_):
    if stop_animation:
        return [img_display]
    
    # Capture a frame
    frame = picam2.capture_array()

    # Convert RGBA to RGB for Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    # Update the displayed image
    img_display.set_data(frame_rgb)
    return [img_display]

# Use Matplotlib's FuncAnimation for real-time updates
ani = FuncAnimation(fig, update, interval=1, cache_frame_data=False)

# Show the video feed
plt.show()

# Cleanup
picam2.stop()
