import cv2
import numpy as np

def get_walkable_tiles(minimap_ss):
    hsv_map = cv2.cvtColor(minimap_ss, cv2.COLOR_BGR2HSV)

    tile_hsvs = {
        "sand_room": np.array([16, 81, 163]),
        "bridge": np.array([12, 98, 120]),
        "player": np.array([0, 0, 255]),
        "ship_room": np.array([170, 61, 63]),
    }

    combined_tiles = np.zeros(hsv_map.shape[:2], dtype=np.uint8)

    tile_masks = {}
    for name, hsv in tile_hsvs.items():
        mask = cv2.inRange(hsv_map, hsv, hsv)
        tile_masks[name] = mask
        combined_tiles = cv2.bitwise_or(combined_tiles, mask)
    
    return combined_tiles

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

    return grid

def resize_print(img, scale):
    return cv2.resize(img, (img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_NEAREST)