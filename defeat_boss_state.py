import numpy as np
from collections import deque
import cv2

from ai_state import AIState
import map_utils
import debug_prints as debug

class DefeatBossState(AIState):
    def __init__(self):
        super().__init__()

    def update(self, ai):
        self._check_debug_toggle()

        minimap_ss = ai.take_minimap_screenshot()

        poi_masks = self._get_poi_masks(minimap_ss)
        combined_unwalkable_mask = self._get_unwalkable_mask(minimap_ss)
        combined_walkable_mask = 255 - combined_unwalkable_mask

        # aim at nearest enemies
        enemies_mask = self._get_enemies_mask(minimap_ss)
        if np.any(enemies_mask):
            self._aim_nearest_enemy(
                enemies_mask, 
                player_loc_rc=ai.MINIMAP_CENTER_RC.copy(), 
                game_region_center_xy=ai.GAME_REGION_CENTER_XY
            )
        combined_poi_mask = self._combine_masks(poi_masks.values(), list(poi_masks.values())[0].shape)

        combined_poi_mask = self._fill_in_center(combined_poi_mask)

        # smooth out map with kernel (mainly used to filter out enemy outlines)
        kernel = np.ones((10, 10), np.uint8)
        combined_poi_mask = self._smooth_out_mask(combined_poi_mask, kernel)
        # DEBUG: Show final mask
        # debug.display_mask("Final Combined", combined_poi_mask)

        nearest_enemy_pt_small_rc, dist = self._get_nearest_reachable_target_rc(enemies_mask, ai.MINIMAP_CENTER_RC, combined_walkable_mask)


        if nearest_enemy_pt_small_rc is not None:
            path, cost = self._get_shortest_path(
                self._downsample_mask(combined_poi_mask),
                start_rc=map_utils.downscale_pt(ai.MINIMAP_CENTER_RC, self.MAP_SHRINK_SCALE), 
                end_rc=nearest_enemy_pt_small_rc
            )
            # debug.display_pathfinding(mask.downsample_mask(map), path, map_utils.downscale_pt(player_loc_rc, self.MAP_SHRINK_SCALE), map_utils.downscale_pt(nearest_enemy_pt_rc, self.MAP_SHRINK_SCALE))

            self._move_along_path(path, steps=self.MOVE_DISTANCE, keypress_duration=ai.KEYPRESS_DURATION)


        # check if boss has been killed
        if not np.all(poi_masks["portal"] == 0):
            self.state_done = True
    
    def _get_nearest_reachable_target_rc(self, enemies_mask, player_loc_rc, combined_walkable_mask):
        enemies_mask_small = self._downsample_mask(enemies_mask)
        poi_mask_small = self._downsample_mask(combined_walkable_mask)

        player_loc_rc_small = map_utils.downscale_pt(player_loc_rc, self.MAP_SHRINK_SCALE)

        visited = set()
        queue = deque([(tuple(player_loc_rc_small), 0)])
        while queue:
            pos, dist = queue.popleft()
            if pos in visited:
                continue
            visited.add(pos)

            if enemies_mask_small[pos[0]][pos[1]] == 255:
                return pos, dist

            for neighbor in self._get_neighbors(pos, poi_mask_small):
                neighbor = tuple(neighbor)
                if neighbor not in visited:
                    queue.append((neighbor, dist+1))
        
        return None, None

    def _get_neighbors(self, pos, small_mask):
        rows, cols = small_mask.shape
        neighbors = []

        for neighbor in [(0,1),(0,-1),(1,0),(-1,0)]:
            index = (pos[0] + neighbor[0], pos[1] + neighbor[1])

            if index[0] < 0 or index[0] >= rows or index[1] < 0 or index[1] >= cols:
                continue

            if small_mask[index[0], index[1]] != 0:
                neighbors.append(index)
        
        return neighbors

    def _get_unwalkable_mask(self, minimap_ss):
        hsv_map = cv2.cvtColor(minimap_ss, cv2.COLOR_BGR2HSV)

        # holds HSV values of poi's
        unwalkable_hsvs = {
            "ship_wall": np.array([17, 114, 170]),
            "crate": np.array([177, 119, 77]),
            "barrel": np.array([0, 107, 105]),
            "clay_barrel": np.array([7, 118, 130]),
            "light_wall": np.array([5, 93, 110]),
            "water": np.array([102, 160, 151]),
            "deep_water": np.array([108, 184, 130]),
            "rocks": np.array([15, 168, 141]),
            "unexplored": np.array([0, 0, 0]),
            "bridge/room": np.array([12, 98, 120])
        }

        unwalkable_masks = {}
        for name, hsv in unwalkable_hsvs.items():
            mask = cv2.inRange(hsv_map, hsv, hsv)
            unwalkable_masks[name] = mask

        combined_mask = np.zeros(list(unwalkable_masks.values())[0].shape, dtype=np.uint8)
        for m in unwalkable_masks.values():
            combined_mask = cv2.bitwise_or(combined_mask, m)
        
        return combined_mask
