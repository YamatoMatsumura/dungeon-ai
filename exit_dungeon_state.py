import numpy as np
import random

from ai_state import AIState
from key_press import press_keys, VK_CODES
import map_utils

class ExitDungeonState(AIState):
    def __init__(self):
        super().__init__()

    def update(self, ai):
        minimap_ss = ai.take_minimap_screenshot()

        poi_masks = self._get_poi_masks(minimap_ss)

        # aim at nearest enemies
        enemies_mask = self._get_enemies_mask(minimap_ss)
        if np.any(enemies_mask):
            self._aim_nearest_enemy(
                enemies_mask, 
                player_loc_rc=ai.MINIMAP_CENTER_RC, 
                game_region_center_xy=ai.GAME_REGION_CENTER_XY
            )
        combined_poi_mask = self._combine_masks(poi_masks.values(), list(poi_masks.values())[0].shape)

        combined_poi_mask = self._fill_in_center(combined_poi_mask)

        # smooth out map with kernel (mainly used to filter out enemy outlines)
        kernel = np.ones((10, 10), np.uint8)
        combined_poi_mask = self._smooth_out_mask(combined_poi_mask, kernel)
        # DEBUG: Show final mask
        # debug.display_mask("Final Combined", combined_poi_mask)

        walkable_mask, walkable_poi_mask = self._get_walkable_pois(combined_poi_mask, poi_masks, ai.MINIMAP_CENTER_RC)

        # check if no walkable spaces
        if np.all(walkable_mask == 0):
            self._fix_no_walkable(walkable_mask, combined_poi_mask, ai.MINIMAP_CENTER_RC)

        # get unstuck if we missed the portal
        if np.all(poi_masks["portal"] == 0):
            self._move_random_direction(1)
            return

        portal_loc_xy = self._get_mask_centers_xy(poi_masks["portal"])
        portal_loc_rc = map_utils.convert_pt_xy_rc(portal_loc_xy[0])
        path, cost = self._get_shortest_path(
            self._downsample_mask(walkable_mask),
            start_rc=map_utils.downscale_pt(ai.MINIMAP_CENTER_RC, self.MAP_SHRINK_SCALE),
            end_rc=map_utils.downscale_pt(portal_loc_rc, self.MAP_SHRINK_SCALE)
        )

        for i in range(len(path) - 3):
            self._move_along_path(path[i:i+3], steps=2, keypress_duration=ai.KEYPRESS_DURATION, scale=self.MAP_SHRINK_SCALE)
            press_keys([VK_CODES['f']])

    def _move_random_direction(duration):
        dir_options = [['w'], ['a'], ['s'], ['d'], ['w', 'a'], ['w', 'd'], ['s', 'a'], ['s', 'd']]
        dir = random.choice(dir_options)

        keys_to_press = []
        for key in dir:
            keys_to_press.append(VK_CODES[key])

        press_keys(keys_to_press, duration=duration)