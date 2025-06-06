import sys
import os
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QWidget, QSlider, QInputDialog, QComboBox
from PyQt5.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QPolygon
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtWidgets import QSizePolicy, QListWidget, QTextEdit, QInputDialog
from PyQt5.QtGui import QImage, QFont
from Clickablebox import ClickableImageLabel
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut
from tools.adjust_video import VideoHandler
from tools.yolo import run_yolo
from tools.logger_config import logger
import pickle
import numpy as np
from reid.reID_inference import compute_homography_distance, hungarian_algorithm
from utils import xyhw_to_xyxy, capture_bbox, extract_bbox_from_label, extract_id_from_label, extract_object_from_label, convert_org_ltwh, ltwh_to_xyxy, convert_source_to_pixmap_coordinate, split_label_string
from reid.homography_wz import get_homography_associator_object
import traceback

class MainWindow(QMainWindow):
    def __init__(self, number_of_views, resolution, parent=None): #conflict
        super(MainWindow, self).__init__(parent)
        self.video_handler_objects = []
        self.number_of_views = int(number_of_views)
        self.video_annotations = {i: {} for i in range(self.number_of_views)} #! to be remained
        logger.info(f"video_annotations: {self.video_annotations}")
        #{view0: {frame0: [bbox0, bbox1, ...], frame1: [bbox0, bbox1, ...], ...}, view1: {frame0: [bbox0, bbox1, ...], frame1: [bbox0, bbox1, ...], ...}, ...}

        self.current_view = 0 #! to be remained
        self.current_frame_index = 0 #! to be remained

        self.setWindowTitle("LabelMV")
        self.setFixedSize(*map(int, resolution.split('x')))
        self.bbox_list_widget = QListWidget() #! list widget
        self.bbox_list_widget.itemDoubleClicked.connect(self.handle_item_double_clicked)
        self.bbox_list_widget.setFixedWidth(300)

        self.text_widget_for_obj = QTextEdit()  # New text widget
        self.text_widget_for_obj.setPlaceholderText("Enter an object")
        self.text_widget_for_obj.setFixedWidth(300)  # Set a fixed height
        self.text_widget_for_obj.setFixedHeight(25)

        self.text_widget_for_id = QTextEdit()  # New text widget
        self.text_widget_for_id.setPlaceholderText("Enter an id")
        self.text_widget_for_id.setFixedWidth(300)  # Set a fixed height
        self.text_widget_for_id.setFixedHeight(25)

        self.objwidget = QTextEdit()  # New text widget
        self.objwidget.setFixedWidth(200)  # Set a fixed height
        self.objwidget.setFixedHeight(25)

        # self.image_list_widget = QListWidget()  # The new QListWidget
        # self.image_list_widget.itemDoubleClicked.connect(self.load_image_from_list) #! to be removed
        # self.image_list_widget.setFixedWidth(200)

        font = QFont()
        font.setPointSize(10) 
        self.frame_indicator = QLabel()
        self.frame_indicator.setText("Current frame: None, Current view: None")  # Initial text
        self.frame_indicator.setFixedHeight(20)
        self.frame_indicator.setFont(font)
        self.frame_indicator.setAlignment(Qt.AlignBottom)
        
        self.resize(1400, 1000)

        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self.browse_video)
        # self.btn_browse.clicked.connect(self.browse_folder)
        self.btn_browse.setFixedWidth(100)

        self.btn_next = QPushButton("Next Frame")
        self.btn_next.clicked.connect(self.next_frame)
        self.btn_next.setFixedWidth(100)
        next_shorcut = QShortcut(QKeySequence('d'), self)
        next_shorcut.activated.connect(self.next_frame)

        self.btn_prev = QPushButton("Previous Frame")
        self.btn_prev.clicked.connect(self.previous_frame)
        self.btn_prev.setFixedWidth(100)
        prev_shorcut = QShortcut(QKeySequence('a'), self)
        prev_shorcut.activated.connect(self.previous_frame)

        self.btn_next_view = QPushButton("Next View")
        self.btn_next_view.clicked.connect(self.show_next_view)
        self.btn_next_view.setFixedWidth(100)
        next_view_shorcut = QShortcut(QKeySequence('w'), self)
        next_view_shorcut.activated.connect(self.show_next_view)

        self.btn_prev_view = QPushButton("Prev View")
        self.btn_prev_view.clicked.connect(self.show_prev_view)
        self.btn_prev_view.setFixedWidth(100)
        prev_view_shorcut = QShortcut(QKeySequence('s'), self)
        prev_view_shorcut.activated.connect(self.show_prev_view)

        self.btn_load_prev_labels = QPushButton("Load prebox")
        self.btn_load_prev_labels.clicked.connect(self.load_prev_labels)  # Connect to the function that runs the YOLO detector
        self.btn_load_prev_labels.setFixedWidth(100)
        load_prev_labels_shortcut = QShortcut(QKeySequence('z'), self)
        load_prev_labels_shortcut.activated.connect(self.load_prev_labels)

        self.btn_clear_all = QPushButton("Clear all")
        self.btn_clear_all.clicked.connect(self.clear_labels)  # Connect to the function that runs the YOLO detector
        self.btn_clear_all.setFixedWidth(100)

        self.btn_run_detector = QPushButton("Run Detector")
        self.btn_run_detector.clicked.connect(self.run_detector)  # Connect to the function that runs the YOLO detector
        self.btn_run_detector.setFixedWidth(100)
        runYolo_shortcut = QShortcut(QKeySequence('q'), self)
        runYolo_shortcut.activated.connect(self.run_detector)

        self.btn_associate_id = QPushButton("Associate IDs")
        self.btn_associate_id.clicked.connect(self.associate_id)
        self.btn_associate_id.setFixedWidth(100)

        self.btn_add_label = QPushButton("Add Label")
        self.btn_add_label.setCheckable(True) 
        self.btn_add_label.clicked.connect(self.add_label)
        self.btn_add_label.setFixedWidth(100)
        add_label_shortcut = QShortcut(QKeySequence('e'), self)
        add_label_shortcut.activated.connect(self.btn_add_label.toggle)

        self.btn_export_label = QPushButton("Export Labels")
        self.btn_export_label.clicked.connect(lambda: self.export_labels(True)) #creates a new function that calls self.export_labels(True) whenever it's called.
        self.btn_export_label.setFixedWidth(100)

        self.btn_import_label = QPushButton("Import Labels")
        self.btn_import_label.clicked.connect(self.import_label)
        self.btn_import_label.setFixedWidth(100)

        self.btn_remove_label = QPushButton("Remove Label")
        self.btn_remove_label.clicked.connect(self.remove_label)
        self.btn_remove_label.setFixedWidth(100)
        remove_label_shortcut = QShortcut(QKeySequence('r'), self)
        remove_label_shortcut.activated.connect(self.remove_label)


        # layout left side
        self.btn_edit_text = QPushButton("Update Box")  # Create the button
        self.btn_edit_text.clicked.connect(self.edit_text)  # Connect it to the function that will handle the button click
        self.btn_edit_text.setFixedWidth(300)  # Set the button width

        self.btn_load_prev_attr = QPushButton("Load Pre-attribute")  # Create the button
        self.btn_load_prev_attr.clicked.connect(self.load_prev_attr)  # Connect it to the function that will handle the button click
        self.btn_load_prev_attr.setFixedWidth(300)  # Set the button width

        self.image_label = ClickableImageLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setPixmap(QPixmap(''))

        self.saved_image_label = QLabel(self)
        self.saved_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.saved_image_label.setMaximumWidth(200)
        self.saved_image_label.setMaximumHeight(200)
        self.saved_image_label.setPixmap(QPixmap(''))

        self.btn_next_id = QPushButton("Next ID")
        self.btn_next_id.clicked.connect(self.next_id)
        self.btn_next_id.setFixedWidth(100)

        self.btn_prev_id = QPushButton("Prev ID")
        self.btn_prev_id.clicked.connect(self.previous_id)
        self.btn_prev_id.setFixedWidth(100)

        self.btn_enter_id = QPushButton("Enter ID") # New text widget
        self.btn_enter_id.clicked.connect(self.enter_id)
        self.btn_enter_id.setFixedWidth(100)

        self.gown_indicator = QLabel("Gown:", self)
        self.gown_indicator.setAlignment(Qt.AlignCenter)
        self.gown_dropdown = QComboBox(self)
        self.gown_dropdown.addItems([
            "NA", "GC", "GI", "GA"
        ])  # Add more options as needed
        gown_layout = QHBoxLayout()
        gown_layout.addWidget(self.gown_indicator)
        gown_layout.addWidget(self.gown_dropdown)

        self.mask_indicator = QLabel("Mask:", self)
        self.mask_indicator.setAlignment(Qt.AlignCenter)
        self.mask_dropdown = QComboBox(self)
        self.mask_dropdown.addItems([
            "NA", "PR", "NC", "RC", "MI", "MA"
        ])  # Add more options as needed
        mask_layout = QHBoxLayout()
        mask_layout.addWidget(self.mask_indicator)
        mask_layout.addWidget(self.mask_dropdown)

        self.eyewear_indicator = QLabel("Eyewear:", self)
        self.eyewear_indicator.setAlignment(Qt.AlignCenter)
        self.eyewear_dropdown = QComboBox(self)
        self.eyewear_dropdown.addItems([
            "NA", "PR", "GG", "FC", "FI", "SG", "PG", "EA"
        ])  # Add more options as needed
        eyewear_layout = QHBoxLayout()
        eyewear_layout.addWidget(self.eyewear_indicator)
        eyewear_layout.addWidget(self.eyewear_dropdown)

        self.GloveL_indicator = QLabel("Glove-L:", self)
        self.GloveL_indicator.setAlignment(Qt.AlignCenter)
        self.GloveL_dropdown = QComboBox(self)
        self.GloveL_dropdown.addItems([
            "NA", "HC", "HA"
        ])  # Add more options as needed
        GloveL_layout = QHBoxLayout()
        GloveL_layout.addWidget(self.GloveL_indicator)
        GloveL_layout.addWidget(self.GloveL_dropdown)

        self.GloveR_indicator = QLabel("Glove-R:", self)
        self.GloveR_indicator.setAlignment(Qt.AlignCenter)
        self.GloveR_dropdown = QComboBox(self)
        self.GloveR_dropdown.addItems([
            "NA", "HC", "HA"
        ])  # Add more options as needed
        GloveR_layout = QHBoxLayout()
        GloveR_layout.addWidget(self.GloveR_indicator)
        GloveR_layout.addWidget(self.GloveR_dropdown)

        # Create a horizontal scrollbar (QSlider)
        self.h_slider = QSlider(Qt.Horizontal)
        self.h_slider.setMinimum(0)
        self.h_slider.setValue(0)
        self.h_slider.setTickPosition(QSlider.TicksBelow)
        self.h_slider.setTickInterval(1)
        self.h_slider.valueChanged.connect(self.update_scroll)

        # Create a QVBoxLayout instance for buttons
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.btn_browse)
        button_layout.addWidget(self.btn_next)
        button_layout.addWidget(self.btn_prev)
        button_layout.addWidget(self.btn_next_view)
        button_layout.addWidget(self.btn_prev_view)
        button_layout.addWidget(self.btn_load_prev_labels)
        button_layout.addWidget(self.btn_run_detector)
        button_layout.addWidget(self.btn_associate_id)
        button_layout.addWidget(self.btn_add_label)
        button_layout.addWidget(self.btn_remove_label)
        button_layout.addWidget(self.btn_clear_all)
        button_layout.addWidget(self.btn_export_label)
        button_layout.addWidget(self.btn_import_label)
        
        # Create a QHBoxLayout for the file label to center it
        frame_indicator_layout = QHBoxLayout()
        frame_indicator_layout.addStretch()
        frame_indicator_layout.addWidget(self.frame_indicator)
        frame_indicator_layout.addStretch()
        
        file_image_layout = QVBoxLayout()
        file_image_layout.addWidget(self.h_slider)  # Move the slider here
        file_image_layout.addLayout(frame_indicator_layout)  # Use the new layout
        file_image_layout.addWidget(self.image_label)
        file_image_layout.setSpacing(0)

        # Create a QVBoxLayout for text and list widgets
        text_list_layout = QVBoxLayout()
        text_list_layout.addWidget(self.text_widget_for_obj)
        text_list_layout.addWidget(self.text_widget_for_id)
        text_list_layout.addLayout(gown_layout)
        text_list_layout.addLayout(mask_layout)
        text_list_layout.addLayout(eyewear_layout)
        text_list_layout.addLayout(GloveL_layout)
        text_list_layout.addLayout(GloveR_layout)
        text_list_layout.addWidget(self.btn_edit_text)
        text_list_layout.addWidget(self.btn_load_prev_attr)
        text_list_layout.addWidget(self.bbox_list_widget)
        # text_list_layout.addWidget(self.image_list_widget)
        
        #The below comment is for image per id loading
        # text_list_layout.addWidget(self.objwidget)
        # text_list_layout.addWidget(self.btn_enter_id)
        # text_list_layout.addWidget(self.btn_next_id)
        # text_list_layout.addWidget(self.btn_prev_id)
        # text_list_layout.addWidget(self.saved_image_label)

        # Create a QHBoxLayout instance for the overall layout
        layout = QHBoxLayout()
        layout.addLayout(button_layout)
        layout.addLayout(file_image_layout)
        layout.addLayout(text_list_layout)

        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)


    def update_scroll(self, value):
        logger.info("<==update_scroll function is called==>")
        logger.info(f"value: {value}")
        self.current_frame_index = value
        self.load_video_frame()


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.previous_image()
        elif event.key() == Qt.Key_Right:
            self.next_image()


    def handle_item_double_clicked(self, item):
        if self.image_label.clicked_rect_index:
            past_index = self.image_label.clicked_rect_index.pop()
            self.image_label.rectangles[past_index]['focus'] = False
            self.image_label.update()
        self.highlight_bbox(item.text())

    
    def highlight_bbox(self, label):
        try:
            bbox, obj, id, attr= split_label_string(label)
            gown, mask, eyewear, gloveL, gloveR = attr.split('-')
            self.text_widget_for_obj.setText(obj)
            self.text_widget_for_id.setText(id)
            self.gown_dropdown.setCurrentIndex(self.gown_dropdown.findText(gown))
            self.mask_dropdown.setCurrentIndex(self.mask_dropdown.findText(mask))
            self.eyewear_dropdown.setCurrentIndex(self.eyewear_dropdown.findText(eyewear))
            self.GloveL_dropdown.setCurrentIndex(self.GloveL_dropdown.findText(gloveL))
            self.GloveR_dropdown.setCurrentIndex(self.GloveR_dropdown.findText(gloveR))

            left, top, width, height = map(int, bbox)
            vertices = [left, top, width, height]
            vertices = xyhw_to_xyxy(vertices)
            right, bottom = vertices[2], vertices[3]

            for i, rect in enumerate(self.image_label.rectangles):
                    if rect['min_xy'] == QPoint(left, top) and rect['max_xy'] == QPoint(right, bottom):

                        self.image_label.rectangles[i]['focus'] = True
                        self.image_label.clicked_rect_index.append(i)
                        break

            self.image_label.update()

        except Exception as e:
            error_details = traceback.format_exc()
            logger.info(f"Exception: {error_details}")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Error:\n{str(e)}\n\nDetails:\n{error_details}")
            msg.setWindowTitle("Warning: hilight bbox")
            msg.exec_()
    


    def export_labels(self, btn=False, sequence_change=False):
        logger.info(f"<==export_labels function is called==>")
        self.video_annotations[self.current_view][self.video_frame_sequences[self.current_frame_index]] = [self.bbox_list_widget.item(i).text() for i in range(self.bbox_list_widget.count())]
        if btn:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getSaveFileName(self, "Save Label File", "", "Text Files (*.txt)", options=options)
            if fileName:
                filename = fileName
            else:
                filename = 'annotations.txt'
        else:
            filename = 'annotations.txt'

        with open(filename, 'w') as f:
            try:
                for view in self.video_annotations:
                    for frame_num, annotations in self.video_annotations[view].items():
                        for i, annotation in enumerate(annotations):
                            bbox, obj, id, attr = split_label_string(annotation)
                            logger.info(f"bbox: {bbox} at export_labels")
                            x, y, w, h = map(int, bbox)
                            org_l, org_t, org_w, org_h  = convert_org_ltwh(x, y, w, h, reverse=False, pixmap=self.pixmap_ref, image_label=self.image_label)

                            f.write(f"{view}, {frame_num}, {id}, {obj}, {attr}, {org_l} {org_t} {org_w} {org_h}\n")

            
            except Exception as e:
                error_details = traceback.format_exc()
                logger.info(f"Exception: {error_details}")
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText(f"Error:\n{str(e)}\n\nDetails:\n{error_details}")
                msg.setWindowTitle("Warning: export label")
                msg.exec_()
    
    
    def enter_id(self):
        self.id = self.objwidget.toPlainText()
        objfolder = f"saved IDs/ID{self.id}"
        if os.path.isdir(objfolder):
            #! self.objimage_files = sorted([f for f in os.listdir(objfolder) if f.endswith(".png")], key=sort_key)
            self.objimage_files = sorted([f for f in os.listdir(objfolder) if f.endswith(".png")])
            if self.objimage_files:  # if there are images in the directory
                self.objcurrent_image_index = 0
                self.load_saved_image(os.path.join(objfolder, self.objimage_files[self.objcurrent_image_index]))


    def next_id(self):
        if self.objimage_files and self.objcurrent_image_index < len(self.objimage_files) - 1:
            # increment the index
            self.objcurrent_image_index += 1
            # load the image
            self.load_saved_image(os.path.join(f"saved IDs/ID{self.id}", self.objimage_files[self.objcurrent_image_index]))
        if self.objcurrent_image_index >= len(self.objimage_files) - 1:
            self.objcurrent_image_index = len(self.objimage_files) - 1
            self.load_saved_image(os.path.join(f"saved IDs/ID{self.id}", self.objimage_files[self.objcurrent_image_index]))


    def previous_id(self):
        if self.objimage_files and self.objcurrent_image_index > 0:
            # increment the index
            self.objcurrent_image_index -= 1
            
            self.load_saved_image(os.path.join(f"saved IDs/ID{self.id}", self.objimage_files[self.objcurrent_image_index]))
        if self.objcurrent_image_index <= 0:
            self.objcurrent_image_index = 0
            self.load_saved_image(os.path.join(f"saved IDs/ID{self.id}", self.objimage_files[self.objcurrent_image_index]))


    def load_saved_image(self, img_path):
        pixmap = QPixmap(img_path)
        scaled_pixmap = pixmap.scaled(self.saved_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.saved_image_label.setPixmap(scaled_pixmap)


    # def load_image_from_list(self, item):
    #     logger.info(f"<==load_image_from_list function is called==>")
    #     self.image_annotations[self.image_files[self.current_image_index]] = [self.bbox_list_widget.item(i).text() for i in range(self.bbox_list_widget.count())]
    #     image_file = item.text()
    #     self.current_image_index = self.image_files.index(image_file)
    #     logger.info(f"image_file: {image_file} (load_image_from_list)")
    #     self.load_video_frame()


    def load_video_frame(self):
        logger.info(f"<==load_video_frame function is called==>")
        self.image_label.clicked_rect_index = []
        sequence = self.video_frame_sequences[self.current_frame_index]
        logger.info(f"video_path_for_views[{self.current_view}]: {self.video_path_for_views[self.current_view]}")
        # pixmap = adjust_video.get_video_frame(self.video_path_for_views[view], sequence)
        pixmap = self.video_handler_objects[self.current_view].get_video_frame(sequence)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio) 
        self.image_label.setPixmap(scaled_pixmap)
        self.frame_indicator.setText(f"Current frame: {sequence} / {self.video_frame_sequences[-1]}, Current view: {self.current_view}")
        self.image_label.rectangles.clear()
        
        # if annotation was made at this frame and at this view
        logger.info(f"annotation: {self.video_annotations} (load_video_frame)")
        logger.info(f"sequence: {sequence} (load_video_frame)")
        if sequence in self.video_annotations[self.current_view]:
            self.bbox_list_widget.clear()
            try: 
                for label in self.video_annotations[self.current_view][sequence]:
                    self.bbox_list_widget.addItem(label)
                    bbox, obj, id, attr = split_label_string(label)
                    logger.info(f"bbox: {bbox}, obj: {obj}, id: {id}, attr: {attr} (load_video_frame)")
                    x, y, w, h = map(int, bbox)
                    rect = {'min_xy': QPoint(x, y), 'max_xy':QPoint(x + w, y + h), 'obj': obj, 'id': id, 'attr': attr, 'focus': False}
                    logger.info(f"rect: {rect} (load_video_frame)")
                    self.image_label.rectangles.append(rect)
           
            except Exception as e:
                error_details = traceback.format_exc()
                logger.info(f"Exception: {error_details}")
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText(f"Error:\n{str(e)}\n\nDetails:\n{error_details}")
                msg.setWindowTitle("Warning: load video frame")
                msg.exec_()
                
        else:
            self.bbox_list_widget.clear()


    def browse_video(self):
        logger.info(f"<==browse_video function is called==>")
        self.video_path_for_views = []
        video_frame_sequences_for_views = []
        self.video_handler_objects = []
        try:
            for i in range(self.number_of_views):
                self.video_path_for_views.append(QFileDialog.getOpenFileName(self, f'Open view {i} Video', '/home')[0])
                self.video_handler_objects.append(VideoHandler(self.video_path_for_views[i]))
            
            fps, ok = QInputDialog.getInt(self, "Set FPS", "Enter desired FPS:", min=1, max=60)
            if ok:
                self.fps = fps
                self.img_size_width_height = self.video_handler_objects[0].get_video_dimensions()
                
                for i in range(self.number_of_views):
                    video_frame_sequences_for_views.append(self.video_handler_objects[i].get_frame_indices(fps))
                    # video_frame_sequences_for_views.append(adjust_video.get_frame_indices(self.video_path_for_views[i], self.fps))
                    logger.info(f"video_frame_sequences_view{i}: {video_frame_sequences_for_views[i]} (browse_video)")

                if len(set(map(len, video_frame_sequences_for_views))) != 1:
                    logger.info(f'Video frame sequences have different lengths')

                self.video_frame_sequences = min(video_frame_sequences_for_views, key=len)

                self.h_slider.setMaximum(len(self.video_frame_sequences) - 1)
                self.current_frame_index = 0
                self.pixmap_ref = self.video_handler_objects[self.current_view].get_video_frame(0)
                self.load_video_frame()
            else:
                QMessageBox.warning(self, "FPS Not Set", "FPS was not set. Please try again.")

        except Exception as e:
            error_details = traceback.format_exc()
            logger.info(f"Exception: {error_details}")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Error: {str(e)}")
            msg.setWindowTitle("Warning: browse video")
            msg.exec_()
        

    def next_frame(self):
        logger.info(f"<==next_frame function is called==>")
        try:
            self.video_annotations[self.current_view][self.video_frame_sequences[self.current_frame_index]] = [self.bbox_list_widget.item(i).text() for i in range(self.bbox_list_widget.count())]
            logger.info(f"annotations: {self.video_annotations} (next_frame)")
            if self.current_frame_index < len(self.video_frame_sequences) - 1:
                self.export_labels(btn=False, sequence_change=True)
                self.current_frame_index += 1
                self.h_slider.setValue(self.current_frame_index)

        except Exception as e:
            error_details = traceback.format_exc()
            logger.info(f"Exception: {error_details}")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Error: {str(e)}")
            msg.setWindowTitle("Warning: next frame")
            msg.exec_()


    def previous_frame(self):
        logger.info(f"<==previous_frame function is called==>")
        try:
            self.video_annotations[self.current_view][self.video_frame_sequences[self.current_frame_index]] = [self.bbox_list_widget.item(i).text() for i in range(self.bbox_list_widget.count())]
            logger.info(f"annotations: {self.video_annotations} (previous_frame)")
            if self.current_frame_index > 0:
                self.export_labels(btn=False, sequence_change=True)
                self.current_frame_index -= 1
                self.h_slider.setValue(self.current_frame_index)
        
        except Exception as e:
            error_details = traceback.format_exc()
            logger.info(f"Exception: {error_details}")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Error: {str(e)}")
            msg.setWindowTitle("Warning: previous frame")
            msg.exec_()


    def show_next_view(self):
        logger.info(f"<==show_next_view function is called==>")
        self.video_annotations[self.current_view][self.video_frame_sequences[self.current_frame_index]] = [self.bbox_list_widget.item(i).text() for i in range(self.bbox_list_widget.count())]
        #try if current view is greater than self.number_of_views
        if self.current_view >= self.number_of_views-1:
            self.current_view = 0
        else:
            self.current_view += 1
        logger.info(f"changed current_view: {self.current_view} (show_second_view)")
        self.load_video_frame()
        self.export_labels()


    def show_prev_view(self):
        logger.info(f"<==show_prev_view function is called==>")
        self.video_annotations[self.current_view][self.video_frame_sequences[self.current_frame_index]] = [self.bbox_list_widget.item(i).text() for i in range(self.bbox_list_widget.count())]
        if self.current_view <= 0:
            self.current_view = self.number_of_views-1
        else:
            self.current_view -= 1
        self.load_video_frame()
        self.export_labels()

    def clear_labels(self):
        logger.info(f"<==clear_labels function is called==>")
        self.bbox_list_widget.clear()
        self.image_label.rectangles.clear()
        self.image_label.update()


    def load_prev_labels(self):
        try:
            logger.info(f"<==load_prev_labels function is called==>")
            prev_sequence = self.video_frame_sequences[self.current_frame_index -1]
            if prev_sequence in self.video_annotations[self.current_view]:
                # self.bbox_list_widget.clear()
                for label in self.video_annotations[self.current_view][prev_sequence]:
                    self.bbox_list_widget.addItem(label)
                    bbox, obj, id, attr = split_label_string(label)
                    x, y, w, h = map(int, bbox)
                    rect = {'min_xy': QPoint(x, y), 'max_xy':QPoint(x + w, y + h), 'obj': obj, 'id': id, 'attr': attr, 'focus': False}
                    self.image_label.rectangles.append(rect)
                self.image_label.update()

        except Exception as e:
            error_details = traceback.format_exc()
            logger.info(f"Exception: {error_details}")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Error: \n{str(e)}\n\nDetails: {error_details}")
            msg.setWindowTitle("Warning: load prebox")
            msg.exec_()

    def import_label(self):
        try:
            logger.info(f"<==import_label function is called==>")
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_name, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Text Files (*.txt)", options=options)
            if file_name:
                with open(file_name, 'r') as f:
                    self.video_annotations.clear()
                    self.video_annotations = {i: {} for i in range(self.number_of_views)}
                    for line in f:
                        view, frame, id, obj, attr, bbox = line.split(', ')
                        x, y, w, h = bbox.split(' ')
                        
                        view, frame = int(view), int(frame)
                        left, top, width, height= convert_org_ltwh(int(x), int(y), int(w), int(h), reverse=True, pixmap=self.pixmap_ref, image_label=self.image_label)

                        if frame not in self.video_annotations[view]:
                            self.video_annotations[view][frame] = [f"({left}, {top}, {width}, {height}), {obj}, {id}, {attr}"]
                        else:
                            self.video_annotations[view][frame].append(f"({left}, {top}, {width}, {height}), {obj}, {id}, {attr}")
                
                self.load_video_frame()

        except Exception as e:
            error_details = traceback.format_exc()
            logger.info(f"Exception: {error_details}")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Error: \n{str(e)}")
            msg.setWindowTitle("Warning: import label")
            msg.exec_()


    def run_detector(self):
        logger.info(f"<==run_detector function is called==>")
        frame = self.video_handler_objects[self.current_view].get_video_frame(self.video_frame_sequences[self.current_frame_index], pixmap=False)
        _, bbox_list = run_yolo(frame)
        self.image_label.rectangles = [] 
        try:
            for bb_left, bb_top, bb_width, bb_height, box_cls in bbox_list:
                
                left, top, width, height = convert_source_to_pixmap_coordinate(bb_left, bb_top, bb_width, bb_height, self.pixmap_ref, self.image_label)
                id, attr = '', ''
                label = f"({left}, {top}, {width}, {height}), {box_cls}, {id}, {attr}"
                existing_items = [self.bbox_list_widget.item(i).text() for i in range(self.bbox_list_widget.count())]
                rect = {"min_xy": QPoint(left, top), "max_xy": QPoint(left + width, top + height), 'obj': box_cls,'id': id, 'attr': attr, 'focus': False}
                
                self.image_label.rectangles.append(rect)

                if label in existing_items:
                    continue  # Skip this bounding box
                
                bbox = f"({left}, {top}, {width}, {height})"

                found = False
                for items in existing_items:
                    if bbox in items:
                        found = True
                        break
                if found:
                    continue

                logger.info(f"generated label: {label}")
                self.bbox_list_widget.addItem(label)

            logger.info(f"annotations: {self.video_annotations}")
            self.image_label.update()

        except Exception as e:
            error_details = traceback.format_exc()
            logger.info(f"Exception: {error_details}")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Error: \n{str(e)}\n\nDetails: {error_details}")
            msg.setWindowTitle("Warning: run detector")
            msg.exec_()
    
    def associate_id(self):
        logger.info(f"<==associate_id function is called==>")
        items = [i for i in range(self.number_of_views)]  # Dropdown options
        items.remove(self.current_view)
        items = [str(i) for i in items]

        selected_view, ok = QInputDialog.getItem(
            self, 
            "Select an option", 
            "Choose a view to be associated:", 
            items, 
            editable=False
        )
        selected_view = int(selected_view)

        try:
            self.video_annotations[self.current_view][self.video_frame_sequences[self.current_frame_index]] = [self.bbox_list_widget.item(i).text() for i in range(self.bbox_list_widget.count())]
            logger.info(f"annotations: {self.video_annotations}")
            if not self.video_annotations[self.current_view] or not self.video_annotations[selected_view]:
                raise ValueError("No bounding box found in source or target view")
            label_list_from_source_view = self.video_annotations[self.current_view][self.video_frame_sequences[self.current_frame_index]]
            label_list_from_target_view = self.video_annotations[selected_view][self.video_frame_sequences[self.current_frame_index]]
            logger.info(f"source view annotations:{self.video_annotations[self.current_view][self.video_frame_sequences[self.current_frame_index]]}")
            logger.info(f"target view annotations:{self.video_annotations[selected_view][self.video_frame_sequences[self.current_frame_index]]}")
            ids_from_source_view = list(map(extract_id_from_label, label_list_from_source_view))
            bbox_list_from_source_view = list(map(extract_bbox_from_label, label_list_from_source_view))
            bbox_list_from_target_view = list(map(extract_bbox_from_label, label_list_from_target_view))
            logger.info(f"bbox_list_from_source_view: {bbox_list_from_source_view}")
            logger.info(f"bbox_list_from_target_view: {bbox_list_from_target_view}")

            associator = get_homography_associator_object()
            bbox_list_from_source_view_xyxy = list(map(ltwh_to_xyxy, bbox_list_from_source_view))
            bbox_list_from_target_view_xyxy = list(map(ltwh_to_xyxy, bbox_list_from_target_view))
            assignments = associator.process_association_above_to_foot(bbox_list_from_source_view_xyxy, bbox_list_from_target_view_xyxy, target_view=selected_view)
            assignments = [[id_from_source_view, pair[1]] for pair, id_from_source_view in zip(assignments, ids_from_source_view)] #change row_index to real id
            assignments = sorted(assignments, key=lambda x: x[1])

            logger.info(f"bbox_list_from_target_view: {bbox_list_from_target_view}")
            new_labels = []
            for (id_source_view, id_target_view), bbox_target_view in zip(assignments, bbox_list_from_target_view):
                attr = ''
                label = f"({bbox_target_view[0]}, {bbox_target_view[1]}, {bbox_target_view[2]}, {bbox_target_view[3]}), person, {id_source_view}, {attr}"
                new_labels.append(label)

            self.video_annotations[selected_view][self.video_frame_sequences[self.current_frame_index]] = new_labels
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)  # Use Information or NoIcon instead of Warning
            msg.setText("Association complete!")
            msg.setWindowTitle("Association Information")  # Adjust title if needed
            msg.exec_()

        except ValueError as e:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(str(e))
            logger.info(f"Value Error: {e}")
            msg.setWindowTitle("Association Warning")
            msg.exec_()

        except Exception as e:
            error_details = traceback.format_exc()
            logger.info(f"Exception: {error_details}")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Error: Not all bboxes have IDs in the source view.")
            msg.setWindowTitle("Warning: associate id")
            msg.exec_()


    def add_label(self):
        logger.info(f"<==add_label function is called==>")
        if self.btn_add_label.isChecked():
            self.image_label.drawing = True
        else:
            self.image_label.drawing = False

    @staticmethod
    def is_convertible_to_int(string):
        try:
            int(string)
            return True
        except ValueError:
            return False


    def remove_label(self):
        logger.info(f"<==remove_label function is called==>")
        try:
            if self.image_label.clicked_rect_index:
                self.image_label.clicked_rect_index.pop()

            item = self.bbox_list_widget.currentItem()
            if item:
                # splited_string = [s.strip() for s in item.text().replace('(', '').replace(')', '').split(',')]
                bbox, obj, id, attr = split_label_string(item.text())

                bbox = list(map(int, bbox))
                coords = xyhw_to_xyxy(bbox)
                rect = {'min_xy': QPoint(coords[0], coords[1]), 'max_xy':QPoint(coords[2], coords[3]), 'obj':obj, 'id': id, 'attr': attr, 'focus':False}
                
                self.bbox_list_widget.takeItem(self.bbox_list_widget.row(item))
                logger.info(f'trying to remove rect: {rect} from rectangle list: {self.image_label.rectangles}')

                if rect in self.image_label.rectangles:
                    self.image_label.rectangles.remove(rect)
                else: #when trying to remove focused bbox, set 'focus' value to True
                    rect['focus'] = True
                    self.image_label.rectangles.remove(rect)

                self.image_label.repaint()

        except Exception as e:
            error_details = traceback.format_exc()
            logger.info(f"Exception: {error_details}")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Error:\n{str(e)}\n\nDetails:\n{error_details}")
            msg.setWindowTitle("Warning: remove label")
            msg.exec_()


    def edit_text(self):
        logger.info(f"<==edit_text function is called==>")
        new_obj = self.text_widget_for_obj.toPlainText()
        new_id = self.text_widget_for_id.toPlainText()
        gown_attr = self.gown_dropdown.currentText()
        mask_attr = self.mask_dropdown.currentText()
        eyewear_attr = self.eyewear_dropdown.currentText()
        gloveL_attr = self.GloveL_dropdown.currentText()
        gloveR_attr = self.GloveR_dropdown.currentText()
        logger.info(f"update new_obj: {new_obj}, new_id: {new_id}, gown_attr: {gown_attr}, mask_attr: {mask_attr}, eyewear_attr: {eyewear_attr}, gloveL_attr: {gloveL_attr}, gloveR_attr: {gloveR_attr}")

        current_item = self.bbox_list_widget.currentItem()   

        try:
            # If an item is selected, update its text
            if current_item is not None:
                current_text = current_item.text()
                bbox, obj, id, attr = split_label_string(current_text)

                # current_item.setText(current_text + ', ' + new_obj + ', ' + new_id)
                current_item.setText(f'({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}), {new_obj}, {new_id}, {gown_attr}-{mask_attr}-{eyewear_attr}-{gloveL_attr}-{gloveR_attr}')
                bbox = list(map(int, bbox))
                vertices = xyhw_to_xyxy(bbox)
                left, top = vertices[0], vertices[1]
                right, bottom = vertices[2], vertices[3]

                # Update the rectangles list with the bounding box ID
                # it has use for loop because whenever you update iamge_label, the paintEvent work same jobs again.
                for i, rect in enumerate(self.image_label.rectangles):
                    if rect['min_xy'] == QPoint(left, top) and rect['max_xy'] == QPoint(right, bottom):

                        logger.info('trying to label bbox class: {}'.format(new_obj))
                        logger.info('trying to label bbox id: {}'.format(new_id))
                        self.image_label.rectangles[i]['obj'] = new_obj
                        self.image_label.rectangles[i]['id'] = new_id
                        self.image_label.rectangles[i]['attr'] = f"{gown_attr}-{mask_attr}-{eyewear_attr}-{gloveL_attr}-{gloveR_attr}"
                        break

            # Force a repaint
            self.image_label.update()

        except Exception as e:
            error_details = traceback.format_exc()
            logger.info(f"Exception: {error_details}")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Error:\n{str(e)}\n\nDetails:\n{error_details}")
            msg.setWindowTitle("Warning: update box")
            msg.exec_()


    def load_prev_attr(self):
        current_item = self.bbox_list_widget.currentItem()   

        try:
            if current_item is not None:
                current_text = current_item.text()
                curr_bbox, curr_obj, curr_id, _ = split_label_string(current_text)

                logger.info(f"<==load_attr_labels function is called==>")
                prev_sequence = self.video_frame_sequences[self.current_frame_index -1]
                if prev_sequence in self.video_annotations[self.current_view]:
                    # self.bbox_list_widget.clear()
                    for label in self.video_annotations[self.current_view][prev_sequence]:
                        # self.bbox_list_widget.addItem(label)
                        _, _, prev_id, prev_attr = split_label_string(label)
                        if prev_id == curr_id:
                            current_item.setText(f'({curr_bbox[0]}, {curr_bbox[1]}, {curr_bbox[2]}, {curr_bbox[3]}), {curr_obj}, {curr_id}, {prev_attr}')
                        # self.image_label.rectangles.append(rect)
                    # self.image_label.update()

        except Exception as e:
            error_details = traceback.format_exc()
            logger.info(f"Exception: {error_details}")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText(f"Error: \n{str(e)}\n\nDetails: {error_details}")
            msg.setWindowTitle("Warning: load preattr")
            msg.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    number_of_views = sys.argv[1] if len(sys.argv) > 1 else "0"
    resolution = sys.argv[2] if len(sys.argv) > 2 else "1280x720"

    main = MainWindow(number_of_views, resolution)
    main.show()

    sys.exit(app.exec_())
