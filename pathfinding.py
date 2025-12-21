import numpy as np
import cv2
from skimage.graph import route_through_array
import mss
import time
from scipy import ndimage
from collections import deque


import debug_prints as debug
from key_press import press_keys, VK_CODES
from globals import Global
import color_mask as mask

MIN_KEYPRESS_DURATION = None
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


def get_boss_heading_xy(game_ss, player_xy):
    template = cv2.imread('sprites/boss_icon.png', cv2.IMREAD_GRAYSCALE)

    game_bgr = np.array(game_ss)[:,:,:3].copy()  # MSS gives a 4th alpha channel, so shrink this down to 3 channels
    game_gray = cv2.cvtColor(game_bgr, cv2.COLOR_BGR2GRAY)  # convert the 3 channels to grayscale

    result = cv2.matchTemplate(game_gray, template, cv2.TM_CCOEFF_NORMED)  

    _, max_val, _, max_loc_xy = cv2.minMaxLoc(result)

    template_height, template_width = template.shape
    template_mid = np.array([max_loc_xy[0] + template_width // 2, max_loc_xy[1] + template_height // 2])

    boss_heading_vec = template_mid - player_xy

    return boss_heading_vec

def get_next_heading(room_vectors, boss_heading, i):

    # If no rooms seen, run towards boss
    if len(room_vectors) == 0:
        return boss_heading

    dots = [np.dot(r/np.linalg.norm(r), boss_heading) for r in room_vectors]
    dots = np.argsort(dots)[::-1]
    return room_vectors[dots[i]]

def get_shortest_path(walkable_mask_small, start_rc, end_rc):

    # walkable_mask has 0 as unwalkable, 255 as walkable
    # 255 marked as true so walkable is 1, 0 is false so unwalkable is np.inf
    cost_array = np.where(walkable_mask_small, 1, 255)


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

def move_along_path(path, steps, scale=1):
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
        press_keys(keys_to_press, duration=Global.MIN_KEYPRESS_DURATION*scale*count)

def parse_new_map(new_walkable_map):
    
    minimap_h, minimap_w = new_walkable_map.shape[:2]
    global_h, global_w = Global.get_map_dim_rc()

    # if first new map...
    if np.all(Global.map == 0):
        # calculate center and subtract off half of new map dims
        start_y = global_h // 2 - minimap_h // 2
        start_x = global_w // 2 - minimap_w // 2

        # add walkable map to center
        Global.map[start_y:start_y+minimap_h, start_x:start_x+minimap_w] = new_walkable_map

        return new_walkable_map, start_x, start_y

    else:
        result = cv2.matchTemplate(Global.map, new_walkable_map, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        start_x, start_y = max_loc    

        region = Global.map[start_y:start_y+minimap_h, start_x:start_x+minimap_w]
        
        # Create mask of true around player region
        player_mask = np.zeros_like(region, dtype=bool)
        minimap_center_h = minimap_h // 2
        minimap_center_w = minimap_w // 2
        player_mask[minimap_center_h - 34 : minimap_center_h + 18, minimap_center_w - 14 : minimap_center_w + 14] = True

        # Around the player, always prefer what we had previously (since force fill walkable around player because of the arrow)
        # debug.display_mask("previous region in global", region)
        processed_region = np.where(player_mask, region, np.maximum(region, new_walkable_map))

        if max_val < 0.85:
            print(f"Confidence of {max_val}, skipping...")

            # Compute rectangle coordinates from player_mask
            rows, cols = np.where(player_mask)
            top_left = (cols.min(), rows.min())        # (x, y)
            bottom_right = (cols.max(), rows.max())    # (x, y)

            # Make copies for visualization (so you don't modify actual mask)
            region_before = cv2.cvtColor(region.copy(), cv2.COLOR_GRAY2BGR)
            region_after = cv2.cvtColor(processed_region.copy(), cv2.COLOR_GRAY2BGR)
            new_walkable = cv2.cvtColor(new_walkable_map.copy(), cv2.COLOR_GRAY2BGR)

            global_color = cv2.cvtColor(Global.map.copy(), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(global_color, (start_x, start_y), (start_x+minimap_w, start_y+minimap_h), (0,0,255), 1)

            # Draw red rectangle (BGR: 0,0,255)
            cv2.rectangle(region_before, top_left, bottom_right, color=(0,0,255), thickness=1)
            cv2.rectangle(region_after, top_left, bottom_right, color=(0,0,255), thickness=1)
            cv2.rectangle(new_walkable, top_left, bottom_right, color=(0,0,255), thickness=1)
            # Display
            cv2.imshow("BAD Stuff previously in Global", region_before)
            cv2.imshow("BAD New walkable map", new_walkable)
            cv2.imshow("BAD Placing new walkable here on global", global_color)
            cv2.imshow("BAD Combined with new map info", region_after)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return processed_region, start_x, start_y

        # add new processed region to our global map
        Global.map[start_y:start_y+minimap_h, start_x:start_x+minimap_w] = processed_region

        # buffer size is the size of the minimap
        buffer_h, buffer_w = minimap_h, minimap_w

        # check if we have one buffer size amount of space around box starting at start_y, start_x
        expand_top = max(0, buffer_h - start_y)
        expand_left = max(0, buffer_w - start_x)
        expand_bottom = max(0, start_y + minimap_h + buffer_h - global_h)
        expand_right = max(0, start_x + minimap_w +buffer_w - global_w)

        if any([expand_top, expand_left, expand_bottom, expand_right]):

            # expand by one buffer size in direction needed
            expand_top_height = (buffer_h if expand_top > 0 else 0)
            expand_bottom_height = (buffer_h if expand_bottom > 0 else 0)
            expand_left_height = (buffer_w if expand_left > 0 else 0)
            expand_right_height = (buffer_w if expand_right > 0 else 0)

            new_h = global_h + expand_top_height + expand_bottom_height
            new_w = global_w + expand_left_height + expand_right_height

            # copy over old map
            new_global = np.zeros((new_h, new_w), dtype=Global.map.dtype)
            new_global[expand_top_height:expand_top_height+global_h, expand_left_height:expand_left_height+global_w] = Global.map

            print(f"Expanded map from {Global.map.shape} to {new_global.shape}")
            Global.map = new_global

            # update global map poi's to match expanded map size
            updated_global_pois = set()
            for poi in Global.poi_pts_xy:
                updated_global_pois.add((poi[0] + expand_left_height, poi[1] + expand_top_height))
            
            Global.poi_pts_xy = updated_global_pois

            return processed_region, start_x + expand_left_height, start_y + expand_top_height

    return processed_region, start_x, start_y

def parse_new_poi(new_poi, max_radius):

    for x, y in Global.poi_pts_xy:

        # if this global poi is nearby the new poi
        if np.linalg.norm((x - new_poi[0], y - new_poi[1])) < max_radius:
            # if this global poi and new poi is rechable within the max radius
            if is_reachable((x, y), new_poi, Global.map, max_radius):
                return
    
    # else, add the new poi to globals
    Global.poi_pts_xy.add(new_poi)

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
    return (int(pt[0] // Global.MAP_SHRINK_SCALE), int(pt[1] // Global.MAP_SHRINK_SCALE))

def convert_pt_to_vec(pt, center):
    return pt - center

def convert_vec_to_pt(vec, center):
    return vec + center

def swap_pt_xy_rc(pt):
    return (pt[1], pt[0])

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