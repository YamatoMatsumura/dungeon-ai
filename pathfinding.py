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

def move_along_path(minimap, scale, indices, steps):

    height, width = minimap.shape[:2]
    player_rc = (height // (2*scale), width // (2*scale))

    # Start at closest index player is at
    closest_index = min(
        range(len(indices)),
        key=lambda i: math.dist(player_rc, indices[i])
    )

    # Make sure only going at max len(indices)
    if steps > len(indices):
        steps = len(indices)

    
    for i in range(1, steps):
        target_rc = indices[closest_index + i]

        dr = target_rc[0] - player_rc[0]
        dc = target_rc[1] - player_rc[1]

        key = map_delta_to_key(dr, dc)

        if key:
            pydirectinput.keyDown(key)
            time.sleep(0.00001)
            pydirectinput.keyUp(key)
        
        player_rc = (player_rc[0] + dr, player_rc[1] + dc)

def update_global_map(global_map, new_walkable_map):

    debug.display_mask("new_walkable", new_walkable_map)
    
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
    
        print(f"found match at: {max_loc}, confidence: {max_val}")
        print(f"Normal start is: {start_x, start_y}")
        # print(f"start_y: {start_y}")
        # print(f"start_x: {start_x}")
        # print(f"start_y + minimap_h - global_h: {start_y + minimap_h - global_h}")
        # print(f"start_x + minimap_w - global_w: {start_x + minimap_w - global_w}")

        global_color = cv2.cvtColor(global_map, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(global_color, (start_x, start_y), (start_x+minimap_w, start_y+minimap_h), (0,0,255), 1)
        cv2.imshow("Overlap", global_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # check if map size needs to be expanded
        expand_top = max(0, -start_y)
        expand_left = max(0, -start_x)
        expand_bottom = max(0, start_y + minimap_h - global_h)
        expand_right = max(0, start_x + minimap_w - global_w)

        if any([expand_top, expand_left, expand_bottom, expand_right]):
            print("Map size changed!")
            input()
            new_h = global_h + expand_top + expand_bottom
            new_w = global_w + expand_left + expand_right

            # copy over old map
            new_global = np.zeroes((new_h, new_w), dtype=global_map.dtype)
            new_global[expand_top:expand_top+global_h, expand_left:expand_left+global_w] = global_map

            # adjust start x and y in cases where needs more space in left or top
            start_x += expand_left
            start_y += expand_top

            global_map = new_global        


        region = global_map[start_y:start_y+minimap_h, start_x:start_x+minimap_w]
        
        # Create mask of true around player region
        player_mask = np.zeros_like(region, dtype=bool)
        minimap_center_h = minimap_h // 2
        minimap_center_w = minimap_w // 2
        player_mask[minimap_center_h - 14 : minimap_center_h + 14, minimap_center_w - 34 : minimap_center_w + 18] = True

        # Around the player, always prefer what we had previously (since force fill walkable around player because of the arrow)
        processed_region = np.where(player_mask, region, np.maximum(region, new_walkable_map))
        global_map[start_y:start_y+minimap_h, start_x:start_x+minimap_w] = processed_region
