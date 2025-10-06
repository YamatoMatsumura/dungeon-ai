import numpy as np
import cv2
from skimage.graph import route_through_array
import pydirectinput
import time
import math

def get_corridor_centroids(minimap_ss):
    
    # isolate out corridors from minimap
    hsv_map = cv2.cvtColor(minimap_ss, cv2.COLOR_BGR2HSV)
    bridge_hsv = np.array([12, 98, 120])
    corridor_mask = cv2.inRange(hsv_map, bridge_hsv, bridge_hsv)

    num_labels, labels = cv2.connectedComponents(corridor_mask)

    centroids = []
    for label in range(1, num_labels):

        mask = (labels == label).astype(np.uint8)
        M = cv2.moments(mask)

        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroids.append(np.array([cx, cy]))

    return centroids

def get_room_centroids(walkable_tiles, minimap_ss):
    # fill in small obstacles
    kernel = np.ones((20,20), np.uint8)
    map_filled = cv2.morphologyEx(walkable_tiles, cv2.MORPH_CLOSE, kernel)

    # filter out corridors by only looking at bigger distances
    dist = cv2.distanceTransform(map_filled.astype(np.uint8), cv2.DIST_L2, 5)
    room_mask = (dist > 13).astype(np.uint8) * 255

    # get connected components
    num_labels, labels = cv2.connectedComponents(room_mask)

    # get center point to filter out spawn centroid
    height, width = minimap_ss.shape[:2]
    spawn_label = labels[height // 2, width // 2]

    centroids = []
    for label in range(1, num_labels):

        if label != spawn_label:
            mask = (labels == label).astype(np.uint8)
            M = cv2.moments(mask)

            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                centroids.append(np.array([cx, cy]))

    # # DEBUG: Draw red circle at centroids
    # # Draw centroids on top
    # map_vis = cv2.cvtColor((map_filled).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # for cx, cy in centroids:
    #     cv2.circle(map_vis, (int(cx), int(cy)), radius=5, color=(0,0,255), thickness=-1)  # red dots

    # cv2.imshow("Walkable Tiles with Centroids", map_vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return centroids

def get_coord_vector_maps(centroids, minimap, coord_to_vec, vec_to_coord):
    height, width = minimap.shape[:2]
    player = np.array([width // 2, height // 2])

    for center in centroids:
        vec = center - player

        # convert to tuple's to use as keys in dict
        coord_tuple = tuple(center)
        vec_tuple = tuple(vec)
        
        coord_to_vec[coord_tuple] = vec_tuple
        vec_to_coord[vec_tuple] = coord_tuple
        # room_vec = room_vec / np.linalg.norm(room_vec)
    
    return coord_to_vec, vec_to_coord

def get_boss_heading(game_ss):
    hsv_map = cv2.cvtColor(game_ss, cv2.COLOR_BGR2HSV)
    height, width = hsv_map.shape[:2]
    player = [width // 2, height // 2]

    hsv_lower = np.array([4, 150, 150])
    hsv_upper = np.array([7, 250, 250])
    mask = cv2.inRange(hsv_map, hsv_lower, hsv_upper)

    # Get moments for image
    M = cv2.moments(mask, binaryImage = True)

    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        average_point = np.array([cx, cy])

        heading_vec = average_point - player
        # heading_vec = heading_vec / np.linalg.norm(heading_vec)
    else:
        heading_vec = None

    return heading_vec

def get_best_room_heading(room_vectors, boss_heading):

    # If no rooms seen, run towards boss
    if len(room_vectors) == 0:
        return boss_heading

    dots = [np.dot(r, boss_heading) for r in room_vectors]
    return room_vectors[np.argmax(dots)]

def get_shortest_path(walkable_tiles_small, minimap_ss, scale, room_vec_to_coord, best_room_vec):
    # establish cost array (walkable has cost 1, not walkable has cost 1000)
    cost_array = np.where(walkable_tiles_small, 1, np.inf)

    height, width = minimap_ss.shape[:2]
    player = (height // (2*scale), width // (2*scale))

    # map room vec to room cord
    best_room_x, best_room_y = room_vec_to_coord[tuple(best_room_vec)]
    end = (int(best_room_y // scale), int(best_room_x // scale))

    # find least cost path to room
    indices, cost = route_through_array(cost_array, start=player, end=end, fully_connected=False)

    if cost == np.inf:
        print("No route")

    return indices

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

def move_along_path(minimap, scale, indices):

    height, width = minimap.shape[:2]
    player = (height // (2*scale), width // (2*scale))

    closest_index = min(
        range(len(indices)),
        key=lambda i: math.dist(player, indices[i])
    )

    target = indices[closest_index + 1]

    dr = target[0] - player[0]
    dc = target[1] - player[1]

    key = map_delta_to_key(dr, dc)

    if key:
        pydirectinput.keyDown(key)
        time.sleep(0.00001)
        pydirectinput.keyUp(key)