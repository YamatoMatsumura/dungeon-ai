import numpy as np
import cv2
from skimage.graph import route_through_array

from scipy import ndimage
from collections import deque

import map_utils
import debug_prints as debug
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
        end_rc = get_nearest_target_rc(walkable_mask_small, end_rc)

    try:
        indices, cost = route_through_array(cost_array, start=start_rc, end=end_rc, fully_connected=True)
        # debug.display_pathfinding(walkable_mask_small, indices, start_rc, end_rc)
        return indices, cost
    except ValueError:
        print("No path found")
        debug.display_pathfinding(walkable_mask_small, indices, start_rc, end_rc)
        return None, None

def get_nearest_target_rc(mask, start_rc):
    # Distance transform does closest zero element for every non-zero element
    # Reverse map since want closest walkable space(255) for every wall(0)
    reversed_mask = 255 - mask
    dist, inds = ndimage.distance_transform_edt(reversed_mask, return_indices = True)

    nearest_r = inds[0, start_rc[0], start_rc[1]]
    nearest_c = inds[1, start_rc[0], start_rc[1]]

    nearest_coord = np.array([nearest_r, nearest_c])
    return nearest_coord

def parse_new_map(new_walkable_map):
    minimap_h, minimap_w = new_walkable_map.shape[:2]

    # if first new map...
    if np.all(Global.current_map == 0):
        Global.current_map = new_walkable_map
    else:
        padded_current_map = map_utils.pad_map(Global.current_map)
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
    center_xy = map_utils.get_center_xy(new_walkable_map)
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