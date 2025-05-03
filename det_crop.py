import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading


class RegionDetectorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Det_crop")
        self.root.geometry("550x480")
        self.setup_gui()

        # Data structures for different region types
        self.current_regions = []  # auto-detected rectangular regions: list of (x, y, w, h)
        self.manual_regions = []  # manually drawn rectangular regions: list of (x, y, w, h)
        self.freehand_polygons = []  # manually drawn freehand polygons: list of lists of (x, y)

        self.drawing_mode = 'rectangle'  # can be 'rectangle' or 'freehand'; by default, rectangle

        # Variables for drawing rectangles
        self.drawing = False
        self.start_point = None

        # Variables for drawing freehand
        self.freehand_drawing = False
        self.temp_polygon_points = []

        # For preview
        self.preview_img = None
        self.original_img = None
        self.scale_factor = 1.0

        # Image path
        self.image_path = None

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.select_btn = ttk.Button(main_frame, text="Select Image", command=self.select_image)
        self.select_btn.pack(pady=10)

        self.file_label = ttk.Label(main_frame, text="No image selected", wraplength=450)
        self.file_label.pack(pady=5)

        # Color threshold adjustment -adjust as req
        threshold_frame = ttk.Frame(main_frame)
        threshold_frame.pack(pady=5)
        ttk.Label(threshold_frame, text="Color Threshold:").pack(side=tk.LEFT)
        self.threshold_var = tk.StringVar(value="200")
        self.threshold_entry = ttk.Entry(threshold_frame, textvariable=self.threshold_var, width=10)
        self.threshold_entry.pack(side=tk.LEFT, padx=5)

        # Size threshold frames
        size_frame = ttk.Frame(main_frame)
        size_frame.pack(pady=5)

        # Minimum size
        ttk.Label(size_frame, text="Min Size:").pack(side=tk.LEFT)
        self.min_size_var = tk.StringVar(value="100")
        self.min_size_entry = ttk.Entry(size_frame, textvariable=self.min_size_var, width=10)
        self.min_size_entry.pack(side=tk.LEFT, padx=5)

        # Maximum size
        ttk.Label(size_frame, text="Max Size:").pack(side=tk.LEFT)
        self.max_size_var = tk.StringVar(value="350")
        self.max_size_entry = ttk.Entry(size_frame, textvariable=self.max_size_var, width=10)
        self.max_size_entry.pack(side=tk.LEFT, padx=5)

        # Rotation angle frame
        rotate_frame = ttk.Frame(main_frame)
        rotate_frame.pack(pady=5)
        ttk.Label(rotate_frame, text="Rotate Angle (°):").pack(side=tk.LEFT)
        self.rotate_var = tk.StringVar(value="0")
        self.rotate_entry = ttk.Entry(rotate_frame, textvariable=self.rotate_var, width=6)
        self.rotate_entry.pack(side=tk.LEFT, padx=5)

        # Rotate button
        self.rotate_btn = ttk.Button(rotate_frame, text="Rotate Image", command=self.rotate_image, state='disabled')
        self.rotate_btn.pack(side=tk.LEFT, padx=5)

        # Preview Button
        self.preview_btn = ttk.Button(main_frame, text="Preview Detection", command=self.preview_detection,
                                      state='disabled')
        self.preview_btn.pack(pady=10)

        # Instructions Label
        instructions_text = (
            "In preview:\n"
            " - Press 'r' to switch to rectangle mode (default).\n"
            " - Press 'f' to switch to freehand mode.\n"
            " - In rectangle mode: left-click & drag to draw a box.\n"
            " - In freehand mode: left-click & drag to draw a freehand polygon.\n"
            " - Press 'c' to clear the last manual region (rectangle or polygon).\n"
            " - Right-click on an auto-detected rectangle to remove it.\n"
            " - Press 'q' to close preview."
        )
        self.instructions_label = ttk.Label(main_frame, text=instructions_text)
        self.instructions_label.pack(pady=5)

        # Process Button
        self.process_btn = ttk.Button(main_frame, text="Process Image", command=self.process_image, state='disabled')
        self.process_btn.pack(pady=10)

        # Status label
        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.pack(pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, length=300, mode='indeterminate')

    def select_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if self.image_path:
            self.file_label.config(text=f"Selected: {os.path.basename(self.image_path)}")
            self.preview_btn.config(state='normal')
            self.rotate_btn.config(state='normal')
            self.status_label.config(text="")
            # Clear previously drawn manual regions
            self.manual_regions = []
            self.current_regions = []
            self.freehand_polygons = []
            # Reset any angle
            self.rotate_var.set("0")

    def rotate_image(self):
        """Rotates the original image by the specified angle, updates self.original_img.Clears existing detections and manual/freehand regions."""
        if not self.image_path:
            return

        try:
            angle = float(self.rotate_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid rotation angle.")
            return

        # Load image if needed
        if self.original_img is None:
            self.original_img = cv2.imread(self.image_path)

        # Perform rotation around the center
        h, w = self.original_img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])

        # Compute new bounding dimensions
        new_w = int((h * sin_val) + (w * cos_val))
        new_h = int((h * cos_val) + (w * sin_val))

        # Adjust rotation matrix to account for translation
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        rotated_img = cv2.warpAffine(self.original_img, rotation_matrix, (new_w, new_h))

        # Update original image
        self.original_img = rotated_img

        # Clear existing detections and regions since the geometry changed
        self.manual_regions = []
        self.current_regions = []
        self.freehand_polygons = []

        # Update status
        self.status_label.config(text=f"Image rotated by {angle}°.")

    def detect_regions(self, image):
        try:
            threshold = int(self.threshold_var.get())
            min_size = int(self.min_size_var.get())
            max_size = int(self.max_size_var.get())
        except ValueError:
            raise ValueError("Please enter valid threshold and size values")

        # Define color bounds
        lower_bound = np.array([0, 0, 0], dtype=np.uint8)
        upper_bound = np.array([threshold, threshold, threshold], dtype=np.uint8)

        # Create and invert mask
        mask = cv2.inRange(image, lower_bound, upper_bound)
        mask = cv2.bitwise_not(mask)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        regions = []
        for i in range(1, num_labels):  # Skip background
            x, y, w, h, area = stats[i]
            if (min_size < w < max_size) and (min_size < h < max_size):
                regions.append((x, y, w, h))
        return regions

    def preview_detection(self):
        try:
            # Read and/or store original image if not already
            if self.original_img is None:
                self.original_img = cv2.imread(self.image_path)

            self.current_regions = self.detect_regions(self.original_img)

            # Create scaled version for display
            self.preview_img = self.scale_image_for_display(self.original_img)

            # Draw auto-detected regions in green with numbers
            for i, (x, y, w, h) in enumerate(self.current_regions):
                sx = int(x * self.scale_factor)
                sy = int(y * self.scale_factor)
                sw = int(w * self.scale_factor)
                sh = int(h * self.scale_factor)
                cv2.rectangle(self.preview_img, (sx, sy),
                              (sx + sw, sy + sh), (0, 255, 0), 2)
                cv2.putText(self.preview_img, str(i + 1), (sx, sy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Show preview with mouse callback
            cv2.namedWindow('Preview - Draw boxes/freehand (r/f), c=clear, q=quit')
            cv2.setMouseCallback('Preview - Draw boxes/freehand (r/f), c=clear, q=quit', self.mouse_callback)

            while True:
                cv2.imshow('Preview - Draw boxes/freehand (r/f), c=clear, q=quit', self.preview_img)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord('c'):
                    # Clear last manual region (rectangle or polygon)
                    if self.freehand_polygons:
                        self.freehand_polygons.pop()
                    elif self.manual_regions:
                        self.manual_regions.pop()
                    # Redraw everything
                    self.redraw_preview()

                elif key == ord('r'):
                    self.drawing_mode = 'rectangle'
                    self.status_label.config(text="Drawing mode: Rectangle")

                elif key == ord('f'):
                    self.drawing_mode = 'freehand'
                    self.status_label.config(text="Drawing mode: Freehand")

            cv2.destroyAllWindows()

            total_regions = len(self.current_regions) + len(self.manual_regions) + len(self.freehand_polygons)
            if total_regions > 0:
                self.process_btn.config(state='normal')
                self.status_label.config(text=f"Found {total_regions} regions total. Ready to process.")
            else:
                self.status_label.config(
                    text="No regions detected or drawn. Try adjusting thresholds or drawing regions.")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def mouse_callback(self, event, x, y, flags, param):
        """Handles both rectangle-mode and freehand-mode drawing, as well as right-click removal
        of auto-detected rectangular regions."""
        # 1. Right-click removal of auto-detected region
        if event == cv2.EVENT_RBUTTONDOWN:
            removed = False
            for idx, (rx, ry, rw, rh) in enumerate(self.current_regions):
                scaled_x = int(rx * self.scale_factor)
                scaled_y = int(ry * self.scale_factor)
                scaled_w = int(rw * self.scale_factor)
                scaled_h = int(rh * self.scale_factor)
                if scaled_x <= x <= scaled_x + scaled_w and scaled_y <= y <= scaled_y + scaled_h:
                    # Remove this region
                    self.current_regions.pop(idx)
                    removed = True
                    self.redraw_preview()
                    break
            if removed:
                return

        # 2. Rectangle drawing mode
        if self.drawing_mode == 'rectangle':
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start_point = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                # Show rectangle as you drag
                img_copy = self.preview_img.copy()
                cv2.rectangle(img_copy, self.start_point, (x, y), (0, 0, 255), 2)
                cv2.imshow('Preview - Draw boxes/freehand (r/f), c=clear, q=quit', img_copy)

            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                x1, y1 = self.start_point
                x2, y2 = x, y
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # Convert coords to original image scale
                scale = self.scale_factor
                orig_x1 = int(x1 / scale)
                orig_y1 = int(y1 / scale)
                orig_x2 = int(x2 / scale)
                orig_y2 = int(y2 / scale)

                # Store the new rectangle
                self.manual_regions.append((orig_x1, orig_y1, orig_x2 - orig_x1, orig_y2 - orig_y1))

                # Redraw preview with the new rectangle
                self.redraw_preview()

        # 3. Freehand drawing mode
        elif self.drawing_mode == 'freehand':
            if event == cv2.EVENT_LBUTTONDOWN:
                self.freehand_drawing = True
                self.temp_polygon_points = [(x, y)]  # start new polygon
            elif event == cv2.EVENT_MOUSEMOVE and self.freehand_drawing:
                # Add current point to the polygon
                self.temp_polygon_points.append((x, y))
                # Draw the freehand line from the last point to the current
                if len(self.temp_polygon_points) > 1:
                    cv2.line(self.preview_img,
                             self.temp_polygon_points[-2],
                             self.temp_polygon_points[-1],
                             (0, 0, 255), 2)
                cv2.imshow('Preview - Draw boxes/freehand (r/f), c=clear, q=quit', self.preview_img)
            elif event == cv2.EVENT_LBUTTONUP:
                self.freehand_drawing = False
                # Close or finalize the freehand polygon
                if len(self.temp_polygon_points) > 2:
                    # Convert scaled coords to original coords
                    scale = self.scale_factor
                    original_polygon = []
                    for (sx, sy) in self.temp_polygon_points:
                        ox = int(sx / scale)
                        oy = int(sy / scale)
                        original_polygon.append((ox, oy))

                    # Store this freehand polygon
                    self.freehand_polygons.append(original_polygon)

                self.redraw_preview()

    def redraw_preview(self):
        """This redraws the preview image from scratch, including the auto-detected rectangular regions (green)
        Manual rectangular regions (red)- Freehand polygons (blue boundary)"""

        self.preview_img = self.scale_image_for_display(self.original_img)

        # Draw auto-detected rectangles in green
        for i, (x, y, w, h) in enumerate(self.current_regions):
            sx = int(x * self.scale_factor)
            sy = int(y * self.scale_factor)
            sw = int(w * self.scale_factor)
            sh = int(h * self.scale_factor)
            cv2.rectangle(self.preview_img, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            cv2.putText(self.preview_img, str(i + 1), (sx, sy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw manually drawn rectangles in red
        for (x, y, w, h) in self.manual_regions:
            sx = int(x * self.scale_factor)
            sy = int(y * self.scale_factor)
            sw = int(w * self.scale_factor)
            sh = int(h * self.scale_factor)
            cv2.rectangle(self.preview_img, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

        # Draw freehand polygons in blue
        for polygon in self.freehand_polygons:
            scaled_polygon = [(int(px * self.scale_factor), int(py * self.scale_factor)) for (px, py) in polygon]
            if len(scaled_polygon) > 1:
                for i in range(len(scaled_polygon) - 1):
                    cv2.line(self.preview_img,
                             scaled_polygon[i],
                             scaled_polygon[i + 1],
                             (255, 0, 0), 2)

        cv2.imshow('Preview - Draw boxes/freehand (r/f), c=clear, q=quit', self.preview_img)

    def scale_image_for_display(self, image, max_height=800):
        height, width = image.shape[:2]
        if height > max_height:
            self.scale_factor = max_height / height
            new_width = int(width * self.scale_factor)
            return cv2.resize(image, (new_width, max_height))
        self.scale_factor = 1.0
        return image.copy()

    def process_image(self):
        # If no regions at all, do nothing
        if not (self.current_regions or self.manual_regions or self.freehand_polygons):
            return

        self.output_folder = filedialog.askdirectory(title="Select Output Folder")
        if not self.output_folder:
            return

        self.select_btn.config(state='disabled')
        self.preview_btn.config(state='disabled')
        self.process_btn.config(state='disabled')
        self.rotate_btn.config(state='disabled')
        self.status_label.config(text="Processing...")
        self.progress.pack(pady=5)
        self.progress.start()

        thread = threading.Thread(target=self.run_processing)
        thread.start()

    def run_processing(self):
        try:
            num_regions = self.crop_regions()
            self.root.after(0, self.processing_complete, num_regions)
        except Exception as e:
            self.root.after(0, self.processing_error, str(e))

    def crop_regions(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Reads the final image
        img = self.original_img

        saved_regions = 0
        source_filename = os.path.splitext(os.path.basename(self.image_path))[0]

        # Process auto-detected rectangles
        for (x, y, w, h) in self.current_regions:
            region = img[y:y + h, x:x + w]
            output_path = os.path.join(self.output_folder, f'{source_filename}_region_{saved_regions + 1}.png')
            cv2.imwrite(output_path, region)
            saved_regions += 1

        # Process manually drawn rectangles
        for (x, y, w, h) in self.manual_regions:
            region = img[y:y + h, x:x + w]
            output_path = os.path.join(self.output_folder, f'{source_filename}_region_{saved_regions + 1}.png')
            cv2.imwrite(output_path, region)
            saved_regions += 1

        # Process freehand polygons
        for polygon in self.freehand_polygons:
            # Create a mask for this polygon
            mask = np.zeros(img.shape[:2], dtype=np.uint8)  # single-channel mask
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

            x, y, w, h = cv2.boundingRect(pts)  # Extract the bounding rectangle of the polygon
            region = img[y:y + h, x:x + w]  # Crop from the original image using the bounding rect

            # Also crop the corresponding area from the mask
            mask_cropped = mask[y:y + h, x:x + w]

            # Create a 3-channel mask if needed
            if len(region.shape) == 3 and region.shape[2] == 3:
                mask_cropped_3ch = cv2.merge([mask_cropped, mask_cropped, mask_cropped])
            else:
                mask_cropped_3ch = mask_cropped

            # Apply the mask so that pixels outside the polygon are black
            region_masked = cv2.bitwise_and(region, mask_cropped_3ch)

            # Save the masked region
            output_path = os.path.join(self.output_folder, f'{source_filename}_region_{saved_regions + 1}_freehand.png')
            cv2.imwrite(output_path, region_masked)
            saved_regions += 1

        return saved_regions

    def processing_complete(self, num_regions):
        self.progress.stop()
        self.progress.pack_forget()
        self.status_label.config(text=f"Successfully cropped {num_regions} regions!")
        self.select_btn.config(state='normal')
        self.preview_btn.config(state='normal')
        self.process_btn.config(state='normal')
        self.rotate_btn.config(state='normal')

    def processing_error(self, error_message):
        self.progress.stop()
        self.progress.pack_forget()
        self.status_label.config(text=f"Error: {error_message}")
        self.select_btn.config(state='normal')
        self.preview_btn.config(state='normal')
        self.process_btn.config(state='normal')
        self.rotate_btn.config(state='normal')

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = RegionDetectorGUI()
    app.run()


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
