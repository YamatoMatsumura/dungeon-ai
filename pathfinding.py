import numpy as np
import cv2
from skimage.graph import route_through_array
import pydirectinput
import time
import math
from scipy import ndimage

import debug_prints as debug

def get_corridor_centroids(corridor_mask):
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

def get_room_centroids(room_mask):
    # get connected components
    num_labels, labels = cv2.connectedComponents(room_mask, connectivity=4)

    # get center point to filter out spawn centroid
    height, width = room_mask.shape[:2]
    spawn_label = labels[height // 2, width // 2]

    centroids = []
    for label in range(1, num_labels):

        if label != spawn_label:
            mask = (labels == label).astype(np.uint8)
            M = cv2.moments(mask)

            if M["m00"] != 0:
                center_x = M["m10"] / M["m00"]
                center_y = M["m01"] / M["m00"]
                centroids.append(np.array([center_x, center_y]))

    # # DEBUG: Draw red circle at centroids
    # # Draw centroids on top
    # map_vis = cv2.cvtColor((map_filled).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # for cx, cy in centroids:
    #     cv2.circle(map_vis, (int(cx), int(cy)), radius=5, color=(0,0,255), thickness=-1)  # red dots

    # cv2.imshow("Walkable Tiles with Centroids", map_vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return centroids

def get_coord_vector_maps(poi_centers_xy, minimap, coord_to_vec, vec_to_coord):
    height, width = minimap.shape[:2]
    player_xy = np.array([width // 2, height // 2])

    for poi_xy in poi_centers_xy:
        poi_vec_xy = poi_xy - player_xy

        # convert to tuple's to use as keys in dict
        coord_tuple = tuple(poi_xy)
        vec_tuple = tuple(poi_vec_xy)
        
        coord_to_vec[coord_tuple] = vec_tuple
        vec_to_coord[vec_tuple] = coord_tuple
        # room_vec = room_vec / np.linalg.norm(room_vec)
    
    return coord_to_vec, vec_to_coord

def get_boss_heading(game_ss):
    template = cv2.imread('sprites/boss_icon.png', cv2.IMREAD_GRAYSCALE)

    screen_height, screen_width = game_ss.shape[:2]
    player_xy = np.array([screen_width // 2, screen_height // 2])

    game_bgr = np.array(game_ss)[:,:,:3].copy()  # MSS gives a 4th alpha channel, so shrink this down to 3 channels
    game_gray = cv2.cvtColor(game_bgr, cv2.COLOR_BGR2GRAY)  # convert the 3 channels to grayscale

    result = cv2.matchTemplate(game_gray, template, cv2.TM_CCOEFF_NORMED)  

    _, max_val, _, max_loc_xy = cv2.minMaxLoc(result)

    template_height, template_width = template.shape
    template_mid = np.array([max_loc_xy[0] + template_width // 2, max_loc_xy[1] + template_height // 2])

    boss_heading_vec = template_mid - player_xy

    return boss_heading_vec

def get_best_room_heading(room_vectors, boss_heading):

    # If no rooms seen, run towards boss
    if len(room_vectors) == 0:
        return boss_heading

    dots = [np.dot(r/np.linalg.norm(r), boss_heading) for r in room_vectors]
    return room_vectors[np.argmax(dots)]

def shrink_walkable_mask(walkable_mask):
    kernel = np.ones((5, 5), np.uint8)
    eroded_walkable_mask = cv2.erode(walkable_mask, kernel)
    return eroded_walkable_mask

def get_shortest_path(walkable_mask_small, minimap_ss, scale, room_vec_to_coord, best_room_vec):

    # walkable_mask has 0 as unwalkable, 255 as walkable
    # 255 marked as true so walkable is 1, 0 is false so unwalkable is np.inf
    cost_array = np.where(walkable_mask_small, 1, np.inf)

    height, width = minimap_ss.shape[:2]
    player_rc = (height // (2*scale), width // (2*scale))

    best_room_x, best_room_y = room_vec_to_coord[best_room_vec]
    end_rc = (int(best_room_y // scale), int(best_room_x // scale))

    # check if current end point is unwalkable
    if walkable_mask_small[end_rc[0], end_rc[1]] == 0:

        # Distance transform does closest zero element for every non-zero element
        # Reverse our map since want closest walkable space(255) for every wall(0)
        reversed_walkable = 255 - walkable_mask_small
        dist, inds = ndimage.distance_transform_edt(reversed_walkable, return_indices=True)

        # inds gives the coordinates of the nearest True (walkable) pixel for every pixel
        nearest_r = inds[0, end_rc[0], end_rc[1]]
        nearest_c = inds[1, end_rc[0], end_rc[1]]
        end_rc = (nearest_r, nearest_c)

    try:
        indices, cost = route_through_array(cost_array, start=player_rc, end=end_rc, fully_connected=False)
        return indices, cost
    except ValueError:
        print("No path found")
        return None, None

def map_delta_to_key(dr, dc):
    if dr == -1 and dc == 0:
        return 'w'
    elif dr == 1 and dc == 0:
        return 's' 
    elif dr == 0 and dc == -1:
        return 'a'  
    elif dr == 0 and dc == 1:
        return 'd'
    else:
        return None  # no movement

def move_along_path(path, steps):

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

        if key and key == last_key:
            count += 1

            if i == steps - 1:
                key_counts.append((last_key, count))
        else:
            if last_key is not None:
                key_counts.append((last_key, count))
            last_key = key
            count = 1
        
    KEY_TIME_MULT = 0.059
    for key, count in key_counts:
        pydirectinput.keyDown(key)
        time.sleep(count*KEY_TIME_MULT)
        pydirectinput.keyUp(key)

def update_global_map(global_map, new_walkable_map):
    
    minimap_h, minimap_w = new_walkable_map.shape[:2]
    global_h, global_w = global_map.shape[:2]

    # if first new map...
    if np.all(global_map == 0):
        # calculate center and subtract off half of new map dims
        start_y = global_h // 2 - minimap_h // 2
        start_x = global_w // 2 - minimap_w // 2

        # add walkable map to center
        global_map[start_y:start_y+minimap_h, start_x:start_x+minimap_w] = new_walkable_map

    else:
        result = cv2.matchTemplate(global_map, new_walkable_map, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        start_x, start_y = max_loc    

        region = global_map[start_y:start_y+minimap_h, start_x:start_x+minimap_w]
        
        # Create mask of true around player region
        player_mask = np.zeros_like(region, dtype=bool)
        minimap_center_h = minimap_h // 2
        minimap_center_w = minimap_w // 2
        player_mask[minimap_center_h - 34 : minimap_center_h + 18, minimap_center_w - 14 : minimap_center_w + 14] = True

        # Around the player, always prefer what we had previously (since force fill walkable around player because of the arrow)
        # debug.display_mask("previous region in global", region)
        processed_region = np.where(player_mask, region, np.maximum(region, new_walkable_map))

        if max_val < 0.85:
            print(f"Bad confidence, skipping...")

            # Compute rectangle coordinates from player_mask
            rows, cols = np.where(player_mask)
            top_left = (cols.min(), rows.min())        # (x, y)
            bottom_right = (cols.max(), rows.max())    # (x, y)

            # Make copies for visualization (so you don't modify actual mask)
            region_before = cv2.cvtColor(region.copy(), cv2.COLOR_GRAY2BGR)
            region_after = cv2.cvtColor(processed_region.copy(), cv2.COLOR_GRAY2BGR)
            new_walkable = cv2.cvtColor(new_walkable_map.copy(), cv2.COLOR_GRAY2BGR)

            global_color = cv2.cvtColor(global_map.copy(), cv2.COLOR_GRAY2BGR)
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
            return

        # add new processed region to our global map
        global_map[start_y:start_y+minimap_h, start_x:start_x+minimap_w] = processed_region


        buffer_h, buffer_w = minimap_h, minimap_w

        # check if we have one buffer size amount of space around box starting at start_y, startx
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
            new_global = np.zeros((new_h, new_w), dtype=global_map.dtype)
            new_global[expand_top_height:expand_top_height+global_h, expand_left_height:expand_left_height+global_w] = global_map

            print(f"Expanded map from {global_map.shape} to {new_global.shape}")
            global_map = new_global

    return global_map