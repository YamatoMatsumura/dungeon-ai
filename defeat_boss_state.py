import numpy as np

from ai_state import AIState
import map_utils

class DefeatBossState(AIState):
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

        nearest_enemy_pt_rc = self._get_nearest_target_rc(enemies_mask, ai.MINIMAP_CENTER_RC)

        path, cost = self._get_shortest_path(
            self._downsample_mask(combined_poi_mask),
            start_rc=map_utils.downscale_pt(ai.MINIMAP_CENTER_RC), 
            end_rc=map_utils.downscale_pt(nearest_enemy_pt_rc)
        )
        # debug.display_pathfinding(mask.downsample_mask(map), path, map_utils.downscale_pt(player_loc_rc), map_utils.downscale_pt(nearest_enemy_pt_rc))

        self._move_along_path(path, steps=self.move_distance)


        # check if boss has been killed
        if not np.all(poi_masks["portal"] == 0):
            self.state_done = True