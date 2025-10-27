import cv2
import numpy as np

import debug_prints as debug

def get_poi_masks(minimap_ss):
    hsv_map = cv2.cvtColor(minimap_ss, cv2.COLOR_BGR2HSV)

    # holds HSV values of poi's
    tile_hsvs = {
        "sand_room": np.array([16, 81, 163]),
        "bridge/room": np.array([12, 98, 120]),
        "player": np.array([0, 0, 255]),
        "ship_room": np.array([170, 61, 63]),
        "carpet": np.array([174, 163, 125]),
        # "enemies": np.array([0, 255, 255]),
        "portal": np.array([120, 255, 255]),
    }

    poi_masks = {}
    for name, hsv in tile_hsvs.items():
        mask = cv2.inRange(hsv_map, hsv, hsv)
        poi_masks[name] = mask
    
    return poi_masks

def combine_masks(masks, shape):
    combined_mask = np.zeros(shape, dtype=np.uint8)
    for m in masks:
        combined_mask = cv2.bitwise_or(combined_mask, m)
    
    return combined_mask

def fill_in_center(mask):
    # Fill in center as walkable since arrow covers it up
    center_r, center_c = mask.shape[0] // 2, mask.shape[1] // 2

    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(
        mask_color,
        (center_c - 14, center_r - 34),
        (center_c + 14, center_r + 18),
        color=255,
        thickness=-1
    )

    return mask

def smooth_out_mask(mask, kernel):
    smoothed_map = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return smoothed_map

def get_room_mask(combined_poi_mask):
    # fill in small obstacles
    kernel = np.ones((20,20), np.uint8)
    map_filled = cv2.morphologyEx(combined_poi_mask, cv2.MORPH_CLOSE, kernel)

    # filter out corridors by only looking at bigger distances
    dist = cv2.distanceTransform(map_filled.astype(np.uint8), cv2.DIST_L2, 5)
    room_mask = (dist > 17).astype(np.uint8) * 255

    return room_mask

def get_walkable_pois(combined_poi_mask, poi_masks):

    # get connected component that is walkable (i.e. the one player is currently located in)
    num_labels, labels = cv2.connectedComponents(combined_poi_mask, connectivity=4)
    
    center_row, center_col = combined_poi_mask.shape[0] // 2, combined_poi_mask.shape[1] // 2
    center_label = labels[center_row, center_col]

    for label in range(1, num_labels):
        if center_label == label:
            walkable_mask = ((labels == label).astype(np.uint8))*255

    # filter down poi mask to only walkable ones
    walkable_poi_mask = {}
    for name, mask in poi_masks.items():
        walkable_poi_mask[name] = (cv2.bitwise_and(mask, walkable_mask))
    
    return walkable_mask, walkable_poi_mask

def downsample_mask(mask, block_size=5):
    """
    mask: boolean array of shape (H, W)
    block_size: number of pixels per grid cell
    """

    H, W = mask.shape
    new_H = H // block_size
    new_W = W // block_size

    # Initialize smaller grid
    grid = np.zeros((new_H, new_W), dtype=bool)

    for i in range(new_H):
        for j in range(new_W):
            block = mask[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            grid[i, j] = np.all(block)  # True if all pixels are walkable

    return grid.astype(np.uint8) * 255