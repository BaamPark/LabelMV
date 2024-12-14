import numpy as np
import os
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt


# def 

def extract_bbox_from_label(label):
    atomic_label = [s.strip() for s in label.replace('(', '').replace(')', '').split(',')]
    bbox = atomic_label[:4]
    bbox = map(int, bbox)
    return list(bbox)

def split_label_string(label: str):
    split_list = label.replace('(', '').replace(')', '').split(', ')
    bbox = split_list[:4]
    obj = split_list[4]
    id = split_list[5]
    attr = split_list[6]
    return bbox, obj, id, attr

def extract_object_from_label(label):
    bbox, obj, id, attr = split_label_string(label)
    return obj

def extract_id_from_label(label):
    bbox, obj, id, attr = split_label_string(label)
    if id == '':
        raise ValueError("current bbox has no id")
    return id


def xyhw_to_xyxy(coords, reverse=False):
    if not reverse:
        coords[2], coords[3] = coords[2] + coords[0], coords[3] + coords[1]
    else:
        coords[2], coords[3] = coords[2] - coords[0], coords[3] - coords[1]
    return coords

def ltwh_to_xyxy(coords, reverse=False):
    if not reverse:
        return [coords[0], coords[1], coords[0] + coords[2], coords[1] + coords[3]]
    else:
        return [coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1]]


#this function will be called when text edit button is pressed
def capture_bbox(bbox, source_path, scale_x, scale_y, vertical_offset, id, frame_num, image_dir):
    import cv2
    # Read the image into a numpy array
    source_image = cv2.imread(source_path)

    # Reverse the scaling and offset
    original_bbox = [int(bbox[0] / scale_x),  # left
                     int((bbox[1] - vertical_offset) / scale_y),  # top
                     int(bbox[2] / scale_x),  # right
                     int((bbox[3] - vertical_offset) / scale_y)]  # bottom
    
    if original_bbox[0] < 0:
        original_bbox[0] = 0
    if original_bbox[1] < 0:
        original_bbox[1] = 0
    if original_bbox[2] > source_image.shape[1]:
        original_bbox[2] = source_image.shape[1]
    if original_bbox[3] > source_image.shape[0]:
        original_bbox[3] = source_image.shape[0]

    # Crop the bounding box from the original image os.path.basename(path)
    bbox_image = source_image[original_bbox[1]:original_bbox[3], original_bbox[0]:original_bbox[2]]

    os.makedirs("saved IDs/ID{}".format(id), exist_ok=True)

    output_path = "saved IDs/ID{}/frame_{}_{}.png".format(id, frame_num, os.path.basename(image_dir))  # replace with your desired output path

    cv2.imwrite(output_path, bbox_image)


def calculate_scale_and_offset(pixmap, image_label):
    image_width = pixmap.width()
    image_height = pixmap.height()

    pixmap = pixmap.scaled(image_label.size(), Qt.KeepAspectRatio)
    scale_x = pixmap.width() / image_width
    scale_y = pixmap.height() / image_height

    vertical_offset = (image_label.height() - pixmap.height()) / 2
    return scale_x, scale_y, vertical_offset


def convert_source_to_pixmap_coordinate(x, y, w, h, pixmap, image_label):
    x, y, w, h = map(int, (x, y, w, h))
    scale_x, scale_y, vertical_offset = calculate_scale_and_offset(pixmap, image_label)
    
    x = int(x * scale_x)
    y = int((y * scale_y) + vertical_offset)
    w = int(w * scale_x)
    h = int(h * scale_y)

    return [x, y, w, h]

def convert_pixmap_to_source_coordinate(x, y, w, h, pixmap, image_label):
    x, y, w, h = map(int, (x, y, w, h))
    scale_x, scale_y, vertical_offset = calculate_scale_and_offset(pixmap, image_label)
    
    x = int(x / scale_x)
    y = int((y - vertical_offset) / scale_y)
    w = int(w / scale_x)
    h = int(h / scale_y)
    
    return [x, y, w, h]


def convert_org_ltwh(bbox0, bbox1, bbox2, bbox3, reverse=False, pixmap=None, image_label=None):
    if not reverse:
        org_ltwh = convert_pixmap_to_source_coordinate(bbox0, bbox1, bbox2, bbox3, pixmap, image_label)
        return org_ltwh[0], org_ltwh[1], org_ltwh[2], org_ltwh[3]
    
    else:
        pix_ltwh = convert_source_to_pixmap_coordinate(bbox0, bbox1, bbox2, bbox3, pixmap, image_label)
        return pix_ltwh[0], pix_ltwh[1], pix_ltwh[2], pix_ltwh[3]

def convert_yolo_format(self, scale_x, scale_y, vertical_offset, bbox0, bbox1, bbox2, bbox3, reverse=False):
    if not reverse:
        org_left = bbox0 / scale_x
        org_top = (bbox1 - vertical_offset) / scale_y
        org_width = bbox2 / scale_x
        org_height = bbox3 / scale_y

        center_x = org_left + org_width / 2
        center_y = org_top + org_height / 2

        yolo_x = center_x / self.img_size_width_height[0]
        yolo_y = center_y / self.img_size_width_height[1]
        yolo_w = org_width / self.img_size_width_height[0]
        yolo_h = org_height / self.img_size_width_height[1]

        return yolo_x, yolo_y, yolo_w, yolo_h
    else:
        center_x = bbox0 * self.img_size_width_height[0]
        center_y = bbox1 * self.img_size_width_height[1]
        org_width = bbox2 * self.img_size_width_height[0]
        org_height = bbox3 * self.img_size_width_height[1]

        center_x = center_x - org_width / 2
        center_y = center_y - org_height / 2

        pix_width = org_width * scale_x
        pix_height = org_height * scale_y
        pix_left = center_x * scale_x
        pix_top = center_y * scale_y + vertical_offset

        return int(pix_left), int(pix_top), int(pix_width), int(pix_height)
    




