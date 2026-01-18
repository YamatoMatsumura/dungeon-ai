import cv2
import numpy as np

from globals import Global

def get_mask_centers_xy(mask):
    num_labels, labels = cv2.connectedComponents(mask, connectivity=4)

    centroids = []
    for label in range(1, num_labels):

        mask = (labels == label).astype(np.uint8)
        M = cv2.moments(mask)

        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            centroids.append(np.array([center_x, center_y]))

    return centroids

def get_corridor_center_xy(corridor_mask):
    num_labels, labels = cv2.connectedComponents(corridor_mask, connectivity=4)

    centroids = []
    for label in range(1, num_labels):

        mask = (labels == label).astype(np.uint8)
        M = cv2.moments(mask)

        if M["m00"] != 0:
            center_x = M["m10"] / M["m00"]
            center_y = M["m01"] / M["m00"]
            centroids.append(np.array([center_x, center_y]))

    return centroids


def get_room_center_xy(room_mask, player_rc):
    # get connected components
    num_labels, labels = cv2.connectedComponents(room_mask, connectivity=4)

    # get center point to filter out spawn centroid
    spawn_label = labels[player_rc[0], player_rc[1]]

    centroids = []
    for label in range(1, num_labels):

        if label != spawn_label:
            mask = (labels == label).astype(np.uint8)
            M = cv2.moments(mask)

            if M["m00"] != 0:
                center_x = M["m10"] / M["m00"]
                center_y = M["m01"] / M["m00"]
                centroids.append(np.array([center_x, center_y]))

    # DEBUG: Draw red circle at centroids
    # Draw centroids on top
    # map_vis = cv2.cvtColor((room_mask).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # for cx, cy in centroids:
    #     cv2.circle(map_vis, (int(cx), int(cy)), radius=5, color=(0,0,255), thickness=-1)  # red dots

    # cv2.imshow("Walkable Tiles with Centroids", map_vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return centroids


def get_boss_heading_vec_xy(game_ss, player_xy):
    template = cv2.imread('sprites/boss_icon.png', cv2.IMREAD_GRAYSCALE)

    game_bgr = np.array(game_ss)[:,:,:3].copy()  # MSS gives a 4th alpha channel, so shrink this down to 3 channels
    game_gray = cv2.cvtColor(game_bgr, cv2.COLOR_BGR2GRAY)  # convert the 3 channels to grayscale

    result = cv2.matchTemplate(game_gray, template, cv2.TM_CCOEFF_NORMED)  

    _, max_val, _, max_loc_xy = cv2.minMaxLoc(result)

    template_height, template_width = template.shape
    template_mid = np.array([max_loc_xy[0] + template_width // 2, max_loc_xy[1] + template_height // 2])

    boss_heading_vec = template_mid - player_xy

    return boss_heading_vec

def pad_map(map):
    height, width = map.shape[:2]

    padded_map = np.zeros((3*height, 3*width), dtype=map.dtype)
    padded_map[height:height*2, width:width*2] = map

    return padded_map

def get_center_rc(map):
    return np.array([map.shape[0] // 2 - 1, map.shape[1] // 2])

def get_center_xy(map):
    return np.array([map.shape[1] // 2 - 1, map.shape[0] // 2])

def downscale_pt(pt):
    return np.array([int(pt[0] // Global.MAP_SHRINK_SCALE), int(pt[1] // Global.MAP_SHRINK_SCALE)])

def convert_pt_to_vec(pt, center):
    return pt - center

def convert_vec_to_pt(vec, center):
    return vec + center

def convert_pt_xy_rc(pt):
    return np.array([pt[1], pt[0]])