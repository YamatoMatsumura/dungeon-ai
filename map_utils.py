import numpy as np

def get_center_rc(map):
    return np.array([map.shape[0] // 2, map.shape[1] // 2])

def get_center_xy(map):
    return np.array([map.shape[1] // 2, map.shape[0] // 2])

def downscale_pt(pt, map_shrink_scale):
    return np.array([int(pt[0] // map_shrink_scale), int(pt[1] // map_shrink_scale)])

def convert_pt_to_vec(pt, center):
    return pt - center

def convert_vec_to_pt(vec, center):
    return vec + center

def convert_pt_xy_rc(pt):
    return np.array([pt[1], pt[0]])