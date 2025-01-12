import gi
import cv2
import numpy as np
from typing import Optional

# Initialize GTK
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf, GLib, Gdk

class DisplayWindow(Gtk.Window):
    """GTK window for displaying camera feeds and detection results.
    
    This class provides a basic display interface that can be used
    for debugging and development purposes.
    """
    
    def __init__(self, title: str = "Camera Feed"):
        """Initialize display window.
        
        Args:
            title: Window title
        """
        Gtk.Window.__init__(self, title=title)
        self.set_default_size(640, 480)
        
        # Create drawing area for efficient display
        self.drawing_area = Gtk.DrawingArea()
        self.add(self.drawing_area)
        self.drawing_area.connect('draw', self._on_draw)
        self.connect("destroy", Gtk.main_quit)
        
        self.current_pixbuf = None
        
    def _on_draw(self, widget, cr):
        """Cairo drawing callback."""
        if self.current_pixbuf is not None:
            Gdk.cairo_set_source_pixbuf(cr, self.current_pixbuf, 0, 0)
            cr.paint()
            
    def update_image(self, frame: np.ndarray) -> None:
        """Update display with new frame.
        
        Args:
            frame: BGR format numpy array
        """
        # Convert BGR to RGB for GTK
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if not frame_rgb.flags['C_CONTIGUOUS']:
            frame_rgb = np.ascontiguousarray(frame_rgb)
            
        height, width, channels = frame_rgb.shape
        
        self.current_pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            frame_rgb.tobytes(),
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