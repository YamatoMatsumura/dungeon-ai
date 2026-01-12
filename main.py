import mss
import numpy as np

import time  # to delay screenshot

import color_mask as mask
import pathfinding
import map_utils
import debug_prints as debug
from globals import Global
import agent_controls as control

time.sleep(2)

control.initialize_pixels_per_step()
control.check_min_duration()

MOVE_DISTANCE_STEPS = 10  # number of steps the player moves between game states
POI_PROXIMITY_RADIUS = 15 # proximity distance two pois can be
POI_VISIT_RADIUS = 20 # proximity distance for a poi to be counted as visited by the player
TARGET_POI_UPDATE_DISTANCE = 30  # if no pois exist within this distance from the current poi, the best aligned one will become the next target

while True:

    with mss.mss() as sct:

        # get minimap screenshot
        minimap_region = {"top": 5, "left": 2032, "width": 522, "height": 533}
        minimap_ss = np.array(sct.grab(minimap_region))

        MINIMAP_CENTER_RC = map_utils.get_center_rc(minimap_ss)
        MINIMAP_CENTER_XY = map_utils.get_center_xy(minimap_ss)

        # get game window screenshot
        game_region = {"top": 0, "left": 0, "width": 2025, "height": 1600}
        game_ss = np.array(sct.grab(game_region))

        GAME_CENTER_XY = map_utils.get_center_xy(game_ss)
        # Debug: Getting minimap pixel region
        # debug.display_BGR(minimap_ss)

        # DEBUG: Get HSV values for stuff on minimap
        # debug.display_HSV(minimap_ss)

    poi_masks = mask.get_poi_masks(minimap_ss)

    # aim at nearest enemies
    enemies_mask = mask.get_enemies_mask(minimap_ss)
    if np.any(enemies_mask):
        control.aim_nearest_enemy(enemies_mask, MINIMAP_CENTER_RC, GAME_CENTER_XY)

    combined_poi_mask = mask.combine_masks(poi_masks.values(), list(poi_masks.values())[0].shape)

    combined_poi_mask = mask.fill_in_center(combined_poi_mask)

    # smooth out map with kernel (mainly used to filter out enemy outlines)
    kernel = np.ones((10, 10), np.uint8)
    combined_poi_mask = mask.smooth_out_mask(combined_poi_mask, kernel)
    # DEBUG: Show final mask
    # debug.display_mask("Final Combined", combined_poi_mask)

    # update current location based on new mask
    pathfinding.parse_new_map(combined_poi_mask)

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

        nearest_walkable_rc = pathfinding.get_nearest_target_rc(combined_poi_mask, start_rc=MINIMAP_CENTER_RC)

        current_map_copy = combined_poi_mask.copy()
        current_map_copy[:, :] = 255  # convert map to all walkable

        path, cost = pathfinding.get_shortest_path(
            current_map_copy,
            start_rc=MINIMAP_CENTER_RC, 
            end_rc=nearest_walkable_rc
        )

        control.move_along_path(path, steps=len(path), slower_movement_adjustment=2)
        continue
 
    if Global.IN_BOSS_ROOM:
        control.shadow_nearest_enemy(enemies_mask, MINIMAP_CENTER_RC, combined_poi_mask, MOVE_DISTANCE_STEPS)

        continue

    poi_pts_xy = []
    poi_relevant_masks = ["bridge/room", "room", "carpet", "ship_room"]
    for relevant_mask in poi_relevant_masks:

        # carpet corresponds to boss room
        boss_check = False
        if relevant_mask == "carpet":
            boss_check = True

        poi_pts_xy.extend(map_utils.get_mask_centers_xy(walkable_poi_mask[relevant_mask], boss_check))


    boss_heading_vec_xy = map_utils.get_boss_heading_vec_xy(game_ss, GAME_CENTER_XY)
    # DEBUG: Display boss heading arrow
    # debug.display_boss_heading(minimap_ss, boss_heading_xy)

    # convert poi pts to vecs
    poi_vec_xy = []
    for pt in poi_pts_xy:
        poi_vec_xy.append(map_utils.convert_pt_to_vec(pt, MINIMAP_CENTER_XY))
    # DEBUG: Display poi vectors
    # debug.display_poi_vectors(minimap_ss, poi_vec_xy)

    # parse new pois to determine if they should be added to global pois
    for pt in poi_pts_xy:
        adjusted_x = int(pt[0] + Global.origin_offset_xy[0])
        adjusted_y = int(pt[1] + Global.origin_offset_xy[1])

        pathfinding.parse_new_poi((adjusted_x, adjusted_y), POI_PROXIMITY_RADIUS)

    # filter out already visited pois
    pathfinding.filter_visited_pois(POI_VISIT_RADIUS)

    # update the global target poi if needed
    if not any(np.array_equal(Global.current_target_pt_xy, p) for p in Global.poi_pts_xy):
        Global.update_target_poi(boss_heading_vec_xy, TARGET_POI_UPDATE_DISTANCE)
    
    debug.display_global_pois(POI_VISIT_RADIUS, TARGET_POI_UPDATE_DISTANCE)

    # shrink map (issue with keypresses can only be so quick, smaller map = less path points returned = more accurate for key press to grid tile)
    walkable_mask_small = mask.downsample_mask(walkable_mask)
    # DEBUG: display smaller map to double check resolution after shrinking
    # debug.display_mask("walkable_mask", walkable_mask)
    # debug.display_mask("downsampled_walkable_mask", debug.resize_print(walkable_mask_small, Global.MAP_SHRINK_SCALE))

    # convert the global current target poi to be with respect to the current map
    adjusted_target_pt_xy = Global.current_target_pt_xy - Global.origin_offset_xy
    
    # convert target poi to rc
    adjusted_target_pt_rc = map_utils.convert_pt_xy_rc(adjusted_target_pt_xy)

    path, cost = pathfinding.get_shortest_path(
        walkable_mask_small, 
        start_rc=map_utils.downscale_pt(MINIMAP_CENTER_RC), 
        end_rc=map_utils.downscale_pt(adjusted_target_pt_rc)
    )

    if cost is not None:
        control.move_along_path(path, steps=10, scale=Global.MAP_SHRINK_SCALE)
    else:
        print("returned cost was None")
        input()
