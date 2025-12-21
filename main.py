import mss
import numpy as np

import time  # to delay screenshot

import color_mask as mask
import pathfinding
import debug_prints as debug
from globals import Global

time.sleep(2)

pathfinding.initialize_pixels_per_step()
pathfinding.check_min_duration()
while True:

    with mss.mss() as sct:

        # get minimap screenshot
        minimap_region = {"top": 5, "left": 2032, "width": 522, "height": 533}
        minimap_ss = np.array(sct.grab(minimap_region))

        MINIMAP_CENTER_RC = pathfinding.get_center_rc(minimap_ss)
        MINIMAP_CENTER_XY = pathfinding.get_center_xy(minimap_ss)

        # get game window screenshot
        game_region = {"top": 0, "left": 0, "width": 2025, "height": 1600}
        game_ss = np.array(sct.grab(game_region))

        GAME_CENTER_XY = pathfinding.get_center_xy(game_ss)
        # Debug: Getting minimap pixel region
        # debug.display_BGR(minimap_ss)

        # DEBUG: Get HSV values for stuff on minimap
        # debug.display_HSV(minimap_ss)

    poi_masks = mask.get_poi_masks(minimap_ss)

    combined_poi_mask = mask.combine_masks(poi_masks.values(), list(poi_masks.values())[0].shape)

    combined_poi_mask = mask.fill_in_center(combined_poi_mask)

    # smooth out map with kernel (mainly used to filter out enemy outlines)
    kernel = np.ones((10, 10), np.uint8)
    combined_poi_mask = mask.smooth_out_mask(combined_poi_mask, kernel)
    # DEBUG: Show final mask
    # debug.display_mask("Final Combined", combined_poi_mask)

    # add new mask to walkable map, as well as update the combined poi mask to handle player arrow
    combined_poi_mask, new_map_start_x, new_map_start_y = pathfinding.parse_new_map(combined_poi_mask)
    # debug.display_mask("Global map", debug.resize_print(global_map, 0.5))

    # rooms done seperatly since looks at distance transform of all poi's instead of pixel values (can't just look at hsv value)
    poi_masks["room"] = mask.get_room_mask(combined_poi_mask)
    # # DEBUG: print out all poi masks
    # for mask_name, poi_mask in poi_masks.items():
    #     debug.display_mask(mask_name, poi_mask)

    # filter down poi's to ones that are reachable from current pos
    walkable_mask, walkable_poi_mask = mask.get_walkable_pois(combined_poi_mask, poi_masks, MINIMAP_CENTER_RC)
    # DEBUG: display walkable masks
    # debug.display_mask("walkable_map", walkable_mask)
    # for mask_name, poi_mask in walkable_poi_mask.items():
    #     debug.display_mask(f"{mask_name}_walkable", poi_mask)

    # check if no walkable spaces
    if np.all(walkable_mask == 0):
        print("Found no walkable spaces")

        nearest_walkable_rc = pathfinding.get_nearest_walkable_rc(combined_poi_mask, start_rc=MINIMAP_CENTER_RC)

        current_map_copy = combined_poi_mask.copy()
        current_map_copy[:, :] = 255  # convert map to all walkable

        path, cost = pathfinding.get_shortest_path(
            current_map_copy,
            start_rc=MINIMAP_CENTER_RC, 
            end_rc=nearest_walkable_rc
        )

        pathfinding.move_along_path(path, steps=len(path))

    poi_pts_xy = []
    poi_pts_xy.extend(pathfinding.get_corridor_center_xy(walkable_poi_mask["bridge/room"]))
    poi_pts_xy.extend(pathfinding.get_room_center_xy(walkable_poi_mask["room"], MINIMAP_CENTER_RC))

    boss_heading_xy = pathfinding.get_boss_heading_xy(game_ss, GAME_CENTER_XY)
    # DEBUG: Display boss heading arrow
    debug.display_boss_heading(minimap_ss, boss_heading_xy)

    # convert poi pts to vecs
    poi_vec_xy = []
    for pt in poi_pts_xy:
        poi_vec_xy.append(pathfinding.convert_pt_to_vec(pt, MINIMAP_CENTER_XY))
    # DEBUG: Display poi vectors
    debug.display_poi_vectors(minimap_ss, poi_vec_xy)

    for pt in poi_pts_xy:
        adjusted_x = int(pt[0] + new_map_start_x)
        adjusted_y = int(pt[1] + new_map_start_y)

        # check whether to add this new poi to global pois based on distance
        pathfinding.parse_new_poi((adjusted_x, adjusted_y), max_radius=15)

    # print(f"global_pois after add: {Global.poi_pts_xy}")
    debug.display_global_pois(Global.map, Global.poi_pts_xy)

    # loop over options in case one poi is unreachable right now
    for i in range(len(poi_pts_xy)):

        # find next best room heading
        best_poi_vec_xy = pathfinding.get_next_heading(poi_vec_xy, boss_heading_xy, i)
        # DEBUG: Display best room vector
        # debug.display_ideal_room(minimap_ss, poi_vec_xy, best_poi_vec_xy)

        # shrink map (issue with keypresses can only be so quick, smaller map = less path points returned = more accurate for key press to grid tile)
        walkable_mask_small = mask.downsample_mask(walkable_mask)
        # DEBUG: display smaller map to double check resolution after shrinking
        # debug.display_mask("walkable_mask", walkable_mask)
        # debug.display_mask("downsampled_walkable_mask", debug.resize_print(walkable_mask_small, Global.MAP_SHRINK_SCALE))

        # convert next heading vec xy to pt
        best_poi_pts_xy = pathfinding.convert_vec_to_pt(best_poi_vec_xy, MINIMAP_CENTER_XY)

        # downscale pt to match smaller map indices
        best_poi_pts_rc = pathfinding.swap_pt_xy_rc(best_poi_pts_xy)

        path, cost = pathfinding.get_shortest_path(
            walkable_mask_small, 
            start_rc=pathfinding.downscale_pt(MINIMAP_CENTER_RC), 
            end_rc=pathfinding.downscale_pt(best_poi_pts_rc)
        )

        if cost is not None:
            pathfinding.move_along_path(path, steps=20, scale=Global.MAP_SHRINK_SCALE)
            break
        elif i == len(poi_pts_xy) - 1:
            debug.display_mask("totally stuck!!", debug.resize_print(walkable_mask_small, 5))