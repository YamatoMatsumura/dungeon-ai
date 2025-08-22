import cv2
import numpy as np

def mask_minimap(minimap_ss):
    hsv_map = cv2.cvtColor(minimap_ss, cv2.COLOR_BGR2HSV)

    # # Debug: Getting HSV for mask
    # plt.imshow(hsv_map, cmap='tab20')
    # plt.show()

    obstacle_hsvs = {
        "rocks": np.array([15, 168, 141]),
        "deep_water": np.array([108, 184, 130]),
        "spawn_trees": np.array([53, 129, 193]),
        "unexplored": np.array([0, 0, 0]),
        "walls": np.array([17, 114, 170])
    }

    combined_obstacles = np.zeros(hsv_map.shape[:2], dtype=np.uint8)

    obstacle_masks = {}
    for name, hsv in obstacle_hsvs.items():
        mask = cv2.inRange(hsv_map, hsv, hsv)
        obstacle_masks[name] = mask
        combined_obstacles = cv2.bitwise_or(combined_obstacles, mask)

    # cv2.imshow("Combined Obstacles", combined_obstacles)
    # cv2.waitKey(0)

    walkable_grid = combined_obstacles == 0  # true/1 = walkable, false/0 = obstacle

    return walkable_grid

def get_boss_heading(boss_ss):
    hsv_map = cv2.cvtColor(boss_ss, cv2.COLOR_BGR2HSV)
    height, width = hsv_map.shape[:2]
    player = [width // 2, height // 2]

    hsv_lower = np.array([4, 150, 150])
    hsv_upper = np.array([7, 250, 250])
    mask = cv2.inRange(hsv_map, hsv_lower, hsv_upper)

    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)

    # Get moments for image
    M = cv2.moments(mask, binaryImage = True)

    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        average_point = np.array([cx, cy])

        heading_vec = average_point - player
        heading_vec = heading_vec / np.linalg.norm(heading_vec)
    else:
        heading_vec = None

    # flip since need it in y,x (row, col)
    heading_vec = heading_vec[::-1]

    return heading_vec

def get_room_headings(minimap_ss):
    hsv_map = cv2.cvtColor(minimap_ss, cv2.COLOR_BGR2HSV)
    height, width = minimap_ss.shape[:2]
    player = [width // 2, height // 2]

    room_hsv = np.array([12, 98, 120])
    mask = cv2.inRange(hsv_map, room_hsv, room_hsv)

    num_labels, labels = cv2.connectedComponents(mask)

    vectors = []
    for i in range(1, num_labels):  # 1 is background label
        ys, xs = np.where(labels == i)
        cx = xs.mean()
        cy = ys.mean()

        # normalize vector
        room = np.array([cx, cy])
        room_vec = room - player
        room_vec = room_vec / np.linalg.norm(room_vec)

        vectors.append(room_vec)
    
    return vectors

def downsample_mask(mask, block_size=4):
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