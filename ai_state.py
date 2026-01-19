from abc import ABC, abstractmethod
import cv2
import numpy as np
import pyautogui
from scipy import ndimage
from skimage.graph import route_through_array


import map_utils
import debug_prints as debug
from key_press import press_keys, VK_CODES

class AIState:
    # serves as the main class that all states will inherit from

    def __init__(self):
        self.MOVE_DISTANCE = 10
        self.MAP_SHRINK_SCALE = 2
        self.state_done = False

    @abstractmethod
    def update(self, ai):
        pass

    def is_done(self):
        return self.state_done

    def _get_poi_masks(self, minimap_ss):
        hsv_map = cv2.cvtColor(minimap_ss, cv2.COLOR_BGR2HSV)

        # holds HSV values of poi's
        tile_hsvs = {
            "sand_room": np.array([16, 81, 163]),
            "bridge/room": np.array([12, 98, 120]),
            "player": np.array([0, 0, 255]),
            # "enemies": np.array([0, 255, 255]),
            "ship_room": np.array([170, 61, 63]),
            "carpet": np.array([174, 163, 125]),
            "portal": np.array([120, 255, 255]),
        }

        poi_masks = {}
        for name, hsv in tile_hsvs.items():
            mask = cv2.inRange(hsv_map, hsv, hsv)
            poi_masks[name] = mask
        
        return poi_masks

    def _get_enemies_mask(self, minimap_ss):
        hsv_map = cv2.cvtColor(minimap_ss, cv2.COLOR_BGR2HSV)
        enemies = np.array([0, 255, 255])

        mask = cv2.inRange(hsv_map, enemies, enemies)
        return mask

    def _aim_nearest_enemy(self, enemies_mask, player_loc_rc, game_region_center_xy):
        # manual player loc adjust since weapon fires a little bit above center
        player_loc_rc[0] += 5

        nearest_enemy_pt_rc = self._get_nearest_target_rc(enemies_mask, player_loc_rc)
        nearest_enemy_pt_xy = map_utils.convert_pt_xy_rc(nearest_enemy_pt_rc)
        player_loc_xy = map_utils.convert_pt_xy_rc(player_loc_rc)
        nearest_enemy_vec_xy = map_utils.convert_pt_to_vec(nearest_enemy_pt_xy, player_loc_xy)

        direction = nearest_enemy_vec_xy / np.linalg.norm(nearest_enemy_vec_xy)
        aim_distance = 300

        aim_loc_xy = game_region_center_xy + direction * aim_distance
        pyautogui.moveTo(aim_loc_xy[0], aim_loc_xy[1])
    
    def _get_nearest_target_rc(self, mask, start_rc):
        # Distance transform does closest zero element for every non-zero element
        # Reverse map since want closest walkable space(255) for every wall(0)
        reversed_mask = 255 - mask
        dist, inds = ndimage.distance_transform_edt(reversed_mask, return_indices = True)

        nearest_r = inds[0, start_rc[0], start_rc[1]]
        nearest_c = inds[1, start_rc[0], start_rc[1]]

        nearest_coord = np.array([nearest_r, nearest_c])
        return nearest_coord

    def _combine_masks(self, masks, shape):
        combined_mask = np.zeros(shape, dtype=np.uint8)
        for m in masks:
            combined_mask = cv2.bitwise_or(combined_mask, m)
        
        return combined_mask

    def _fill_in_center(self, mask):
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

    def _smooth_out_mask(self, mask, kernel):
        smoothed_map = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return smoothed_map

    def _get_walkable_pois(self, combined_poi_mask, poi_masks, player_rc):

        # get connected component that is walkable (i.e. the one player is currently located in)
        num_labels, labels = cv2.connectedComponents(combined_poi_mask, connectivity=4)
        
        player_r, player_c = player_rc
        center_label = labels[player_r, player_c]

        # create overall walkable mask
        walkable_mask = np.zeros_like(labels, dtype=np.uint8)
        for label in range(1, num_labels):
            if center_label == label:
                walkable_mask = ((labels == label).astype(np.uint8))*255

        # filter down poi mask to only walkable ones
        walkable_poi_mask = {}
        for name, mask in poi_masks.items():
            walkable_poi_mask[name] = (cv2.bitwise_and(mask, walkable_mask))
        
        return walkable_mask, walkable_poi_mask
    
    def _fix_no_walkable(self, combined_poi_mask, player_loc_rc, keypress_duration):

        nearest_walkable_rc = self._get_nearest_target_rc(combined_poi_mask, start_rc=player_loc_rc)

        current_map_copy = combined_poi_mask.copy()
        current_map_copy[:, :] = 255  # convert map to all walkable

        path, cost = self._get_shortest_path(
            current_map_copy,
            start_rc=player_loc_rc, 
            end_rc=nearest_walkable_rc
        )

        self._move_along_path(path, steps=len(path), keypress_duration=keypress_duration, slower_movement_adjustment=2)

    def _get_shortest_path(self, walkable_mask_small, start_rc, end_rc):

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
            end_rc = self._get_nearest_target_rc(walkable_mask_small, end_rc)

        try:
            indices, cost = route_through_array(cost_array, start=start_rc, end=end_rc, fully_connected=True)
            # debug.display_pathfinding(walkable_mask_small, indices, start_rc, end_rc)
            return indices, cost
        except ValueError:
            print("No path found")
            debug.display_pathfinding(walkable_mask_small, indices, start_rc, end_rc)
            return None, None
    
    def _move_along_path(self, path, steps, keypress_duration, scale=1, slower_movement_adjustment=1):
        # Make sure only going at max len(path)
        if steps >= len(path):
            steps = len(path) - 1
        
        key_counts = []
        last_key = None
        count = 0
        for i in range(steps):
            dr = path[i+1][0] - path[i][0]
            dc = path[i+1][1] - path[i][1]

            key = self._map_delta_to_key(dr, dc)
            
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
            press_keys(keys_to_press, duration=keypress_duration*count*scale*slower_movement_adjustment)
    
    def _map_delta_to_key(self, dr, dc):
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

    def _downsample_mask(self, mask):
        # Find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # create new mask with shrunken dimensions
        new_H = mask.shape[0] // self.MAP_SHRINK_SCALE
        new_W = mask.shape[1] // self.MAP_SHRINK_SCALE
        downsampled_mask = np.zeros((new_H, new_W), dtype=np.uint8)

        for i, cnt in enumerate(contours):
            cnt_scaled = cnt // self.MAP_SHRINK_SCALE

            # if this contours parent is -1, top level contour so fill normally
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(downsampled_mask, [cnt_scaled], -1, color=255, thickness=-1)
            
            # else, indicates a whole within the contour, so fill with 0's
            else:
                cv2.drawContours(downsampled_mask, [cnt_scaled], -1, color=0, thickness=-1)
        
        return downsampled_mask

    def _get_mask_centers_xy(self, mask):
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