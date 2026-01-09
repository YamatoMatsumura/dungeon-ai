import numpy as np
import cv2
from skimage.graph import route_through_array
import mss
import time
from scipy import ndimage
from collections import deque

import map_transforms
import debug_prints as debug
from key_press import press_keys, VK_CODES
from globals import Global

MIN_KEYPRESS_DURATION = None

def get_shortest_path(walkable_mask_small, start_rc, end_rc):

    # erode map to account for slight inaccuracies in movement
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded_walkable = cv2.erode(walkable_mask_small, kernel)

    # walkable_mask has 0 as unwalkable, 255 as walkable
    # 255 marked as true so walkable is 1, 0 is false so unwalkable is np.inf
    cost_array = np.where(eroded_walkable, 1, 255)

    # ensure end_rc is within bounds of the map
    h, w = walkable_mask_small.shape
    end_rc = np.clip(end_rc, [0, 0], [h - 1, w - 1])

    # check if current end point is unwalkable
    if walkable_mask_small[end_rc[0], end_rc[1]] == 0:
        end_rc = get_nearest_walkable_rc(walkable_mask_small, end_rc)

    try:
        indices, cost = route_through_array(cost_array, start=start_rc, end=end_rc, fully_connected=True)
        debug.display_pathfinding(walkable_mask_small, indices, start_rc, end_rc)
        return indices, cost
    except ValueError:
        print("No path found")
        debug.display_pathfinding(walkable_mask_small, indices, start_rc, end_rc)
        return None, None

def get_nearest_walkable_rc(mask, start_rc):
    # Distance transform does closest zero element for every non-zero element
    # Reverse map since want closest walkable space(255) for every wall(0)
    reversed_mask = 255 - mask
    dist, inds = ndimage.distance_transform_edt(reversed_mask, return_indices = True)

    nearest_r = inds[0, start_rc[0], start_rc[1]]
    nearest_c = inds[1, start_rc[0], start_rc[1]]

    nearest_coord = np.array([nearest_r, nearest_c])
    return nearest_coord

def map_delta_to_key(dr, dc):
    if dr == -1 and dc == 0:
        return ['w']
    elif dr == 1 and dc == 0:
        return ['s'] 
    elif dr == 0 and dc == -1:
        return ['a']  
    elif dr == 0 and dc == 1:
        return ['d']
    elif dr == -1 and dc == -1:
        return ['w', 'a']
    elif dr == -1 and dc == 1:
        return ['w', 'd']
    elif dr == 1 and dc == -1:
        return ['s', 'a']
    elif dr == 1 and dc == 1:
        return ['s', 'd']
    else:
        return None  # no movement

def move_along_path(path, steps, scale=1, slower_movement_adjustment=1):
    # Make sure only going at max len(path)
    if steps > len(path):
        steps = len(path)
    
    key_counts = []
    last_key = None
    count = 0
    for i in range(1, steps):
        dr = path[i][0] - path[i-1][0]
        dc = path[i][1] - path[i-1][1]

        key = map_delta_to_key(dr, dc)
        
        # initialize last_key on the first one
        if last_key == None:
            last_key = key
            count += 1
        # add 1 to key count if same key from last key
        elif key == last_key:
            count += 1
        # else, key != last_key, so add the current key and count to key_counts and start counting this next one
        else:
            key_counts.append((last_key, count))
            last_key = key
            count = 1
        
        # make sure last iteration adds the current key and count to key_count
        if i == steps - 1:
            key_counts.append((last_key, count))

    for key, count in key_counts:
        keys_to_press = []
        for k in key:
            keys_to_press.append(VK_CODES[k])
        press_keys(keys_to_press, duration=Global.MIN_KEYPRESS_DURATION*count*scale*slower_movement_adjustment)

def parse_new_map(new_walkable_map):
    minimap_h, minimap_w = new_walkable_map.shape[:2]

    # if first new map...
    if np.all(Global.current_map == 0):
        Global.current_map = new_walkable_map
    else:
        padded_current_map = map_transforms.pad_map(Global.current_map)
        result = cv2.matchTemplate(padded_current_map, new_walkable_map, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # safety check for low template matching confidence
        if max_val < 0.85:
            print(f"Confidence of {max_val} is bad...")
            input()

        # calculate how much we moved
        start_x, start_y = max_loc
        moved_x = start_x - minimap_w
        moved_y = start_y - minimap_h
        Global.origin_offset_xy += np.array([moved_x, moved_y])

        # around the player, always prefer what we had previously
        center_h = minimap_h // 2
        center_w = minimap_w // 2
        arrow_top = 34
        arrow_bot = 18
        arrow_left = 14
        arrow_right = 14
        new_walkable_map[
            center_h - arrow_top : center_h + arrow_bot,
            center_w - arrow_left : center_w + arrow_right
        ] = Global.current_map[
            center_h - arrow_top + moved_y : center_h + arrow_bot + moved_y,
            center_w - arrow_left + moved_x : center_w + arrow_right + moved_x
        ]

        # update the current map
        Global.current_map = new_walkable_map

    # add current loc to visited set
    center_xy = get_center_xy(new_walkable_map)
    Global.visited_xy.append(tuple(center_xy + Global.origin_offset_xy))

def parse_new_poi(new_poi, poi_proximity_radius):

    # make sure new poi isn't too close to an existing poi
    for existing_poi in Global.poi_pts_xy:
        distance = np.linalg.norm([existing_poi[0] - new_poi[0], existing_poi[1] - new_poi[1]])
        if distance < poi_proximity_radius:
                return

    # else, add the new poi to globals
    Global.poi_pts_xy.append(new_poi)

def filter_visited_pois(poi_visited_radius):
    global_pois_copy = Global.poi_pts_xy.copy()
    for poi_xy in global_pois_copy:
        for visited_xy in Global.visited_xy:
            distance = np.linalg.norm([poi_xy[0] - visited_xy[0], poi_xy[1] - visited_xy[1]])
            if distance < poi_visited_radius:
                Global.poi_pts_xy.remove(poi_xy)
                break

def is_reachable(source, target, global_map, max_radius=5): 
    queue = deque()
    queue.append((source[0], source[1], 0))

    visited = set()
    visited.add(source)
    directions = [(0,1), (0,-1), (1,0), (-1,0)]

    max_rows, max_cols = global_map.shape

    while queue:
        x, y, distance = queue.popleft()

        # skip point if too far from source point
        if distance > max_radius:
            continue

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # skip if point is not within bounds
            if not (0 <= nx < max_cols and 0 <= ny < max_rows):
                continue

            # skip if point is not walkable
            if global_map[ny, nx] == 0:
                continue

            # skip if already visited
            if (nx, ny) in visited:
                continue

            if (nx, ny) == target:
                return True

            visited.add((nx, ny))
            queue.append((nx, ny, distance+1))

    return False

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

def initialize_pixels_per_step():
    lower_bound = None
    upper_bound = None

    keypress_duration = 0.001  # assumed to safely return a pixel offset of 0
    while lower_bound is None or upper_bound is None:
        with mss.mss() as sct:

            # get minimap screenshot
            minimap_region = {"top": 5, "left": 2032, "width": 522, "height": 533}
            initial_ss = np.array(sct.grab(minimap_region))

            # pad initial screenshot so template matching works
            pad = 20
            expanded_initial = cv2.copyMakeBorder(
                initial_ss,
                pad, pad, pad, pad,
                borderType=cv2.BORDER_REPLICATE
            )

            time.sleep(0.01)
            press_keys([VK_CODES["w"]], duration=keypress_duration)
            time.sleep(0.01)
            moved_ss = np.array(sct.grab(minimap_region))

            result = cv2.matchTemplate(expanded_initial, moved_ss, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            time.sleep(0.01)
            press_keys([VK_CODES["s"]], duration=keypress_duration)
            time.sleep(0.01)
            pixels_per_step = pad - max_loc[1]

            if pixels_per_step == 1 and lower_bound is None:
                lower_bound = keypress_duration
            elif pixels_per_step > 1 and upper_bound is None:
                upper_bound = keypress_duration
            else:
                keypress_duration += 0.001
            
            time.sleep(0.01)
    
    Global.MIN_KEYPRESS_DURATION = (lower_bound + 0.25*(upper_bound + lower_bound))  # bias towards lower end of bound
    print(f"Found min keypress was {Global.MIN_KEYPRESS_DURATION}")

def check_min_duration():
    with mss.mss() as sct:
        # get minimap screenshot
        minimap_region = {"top": 5, "left": 2032, "width": 522, "height": 533}
        initial_ss = np.array(sct.grab(minimap_region))

        # pad initial screenshot so template matching works
        pad = 20
        expanded_initial = cv2.copyMakeBorder(
            initial_ss,
            pad, pad, pad, pad,
            borderType=cv2.BORDER_REPLICATE
        )

        time.sleep(0.01)
        press_keys([VK_CODES["w"]], duration=Global.MIN_KEYPRESS_DURATION*2)
        time.sleep(0.01)
        moved_ss = np.array(sct.grab(minimap_region))

        result = cv2.matchTemplate(expanded_initial, moved_ss, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        time.sleep(0.01)
        press_keys([VK_CODES["s"]], duration=Global.MIN_KEYPRESS_DURATION*2)
        time.sleep(0.01)
        pixels_per_step = pad - max_loc[1]

        print(f"pixels per step: {pixels_per_step}")