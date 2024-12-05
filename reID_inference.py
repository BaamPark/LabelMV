import pickle
import numpy as np

homography = pickle.load(open('homography.pkl', 'rb'))
homography_inv = pickle.load(open('homography_inv.pkl', 'rb'))

def map_bbox_with_homography(left: int, top: int, bottom: int, right: int): 
    homography = pickle.load(open('homography.pkl', 'rb'))
    center_x = int((left + right) / 2)
    target_point = [center_x, bottom]

    target_point_homogeneous = np.array([*target_point, 1])
    mapped_target_point = np.dot(homography, target_point_homogeneous)[:2]