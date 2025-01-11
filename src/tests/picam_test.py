from picamera2 import Picamera2
import cv2
import gi
import time
import numpy as np

# Initialize GTK
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf, GLib, Gdk

class CameraWindow(Gtk.Window):
    def __init__(self):
        print("Initializing window...")
        Gtk.Window.__init__(self, title="Camera Feed")
        self.set_default_size(640, 480)
        
        # Use DrawingArea for efficient display
        self.drawing_area = Gtk.DrawingArea()
        self.add(self.drawing_area)
        self.drawing_area.connect('draw', self.on_draw)
        self.connect("destroy", Gtk.main_quit)
        
        # Initialize frame buffer
        self.current_pixbuf = None
        print("Window initialized")
        
    def on_draw(self, widget, cr):
        if self.current_pixbuf is not None:
            Gdk.cairo_set_source_pixbuf(cr, self.current_pixbuf, 0, 0)
            cr.paint()
        
    def update_image(self, frame):
        """Update display with new frame data.
        
        Args:
            frame: Frame data in RGB format required by GTK
        """
        height, width, channels = frame.shape
        
        # Ensure frame data is contiguous
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
            
        # Create pixbuf from RGB frame data
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
        
        # Force redraw
        self.drawing_area.queue_draw()

def main():
    # Initialize camera
    picam2 = Picamera2()
    # Configure camera for RGB888 format (which actually gives BGR-ordered pixels)
    camera_config = picam2.create_preview_configuration(
        main={"format": 'RGB888', "size": (640, 480)}
    )
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(2)  # Allow camera to initialize

    # Create and show window
    window = CameraWindow()
    window.show_all()

    def update_display():
        # Capture new frame (comes in BGR format from RGB888 config)
        frame = picam2.capture_array()
        
        # Convert from BGR to RGB for GTK display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update display with RGB frame
        window.update_image(frame_rgb)
        
        # Process any pending GTK events
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
        picam2.stop()
        Gtk.main_quit()

if __name__ == "__main__":
    main()