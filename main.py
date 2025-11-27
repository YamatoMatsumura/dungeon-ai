import mss
import numpy as np

import time  # to delay screenshot

import color_mask as mask
import pathfinding
import debug_prints as debug

time.sleep(2)

global_map = np.zeros((1000, 1000), dtype=np.uint8)
MAP_SHRINK_SCALE = 5
while True:

    with mss.mss() as sct:

        # get minimap screenshot
        minimap_region = {"top": 5, "left": 2032, "width": 522, "height": 533}
        minimap_ss = np.array(sct.grab(minimap_region))

        # get game window screenshot
        game_region = {"top": 0, "left": 0, "width": 2025, "height": 1600}
        game_ss = np.array(sct.grab(game_region))
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
    global_map, combined_poi_mask = pathfinding.parse_new_map(global_map, combined_poi_mask)
    debug.display_mask("Global map", debug.resize_print(global_map, 0.5))

    # rooms done seperatly since looks at distance transform of all poi's instead of pixel values (can't just look at hsv value)
    poi_masks["room"] = mask.get_room_mask(combined_poi_mask)
    # DEBUG: print out all poi masks
    # for mask_name, poi_mask in poi_masks.items():
    #     debug.display_mask(mask_name, poi_mask)

    # filter down poi's to ones that are reachable from current pos
    walkable_mask, walkable_poi_mask = mask.get_walkable_pois(combined_poi_mask, poi_masks)
    # DEBUG: display walkable masks
    # debug.display_mask("walkable_map", walkable_mask)
    # for mask_name, poi_mask in walkable_poi_mask.items():
    #     debug.display_mask(f"{mask_name}_walkable", poi_mask)

    # check if no walkable spaces
    if np.all(walkable_mask == 0):
        print("Found no walkable")
        center_row, center_col = combined_poi_mask.shape[0] // 2, combined_poi_mask.shape[1] // 2
        nearest_walkable_rc = pathfinding.get_nearest_walkable(combined_poi_mask, (center_row, center_col))

        map_small = mask.downsample_mask(combined_poi_mask, MAP_SHRINK_SCALE)
        map_small[:, :] = 255
        start_rc = (center_row // MAP_SHRINK_SCALE, center_col // MAP_SHRINK_SCALE)
        nearest_walkable_rc = (nearest_walkable_rc[0] // MAP_SHRINK_SCALE, nearest_walkable_rc[1] // MAP_SHRINK_SCALE)
        path, cost = pathfinding.get_shortest_path(map_small, start_rc=start_rc, end_rc=nearest_walkable_rc)
        if cost is not None:
            pathfinding.move_along_path(path, steps=12)
        elif i == len(poi_coord_to_vec) - 1:
            debug.display_mask("totally stuck!!", debug.resize_print(walkable_mask_small, 5))

    corridor_centroids = pathfinding.get_corridor_centroids(walkable_poi_mask["bridge/room"])
    room_centroids = pathfinding.get_room_centroids(walkable_poi_mask["room"])

    poi_coord_to_vec = {}
    poi_vec_to_coord = {}

    # add corridor centroids to vec coord mappings
    poi_coord_to_vec, poi_vec_to_coord = pathfinding.get_coord_vector_maps(corridor_centroids, minimap_ss, poi_coord_to_vec, poi_vec_to_coord)
    # add room centroids to vec coord mappings
    poi_coord_to_vec, poi_vec_to_coord = pathfinding.get_coord_vector_maps(room_centroids, minimap_ss, poi_coord_to_vec, poi_vec_to_coord)
    # DEBUG: Display poi vectors
    # debug.display_poi_vectors(minimap_ss, poi_vec_to_coord)

    boss_heading = pathfinding.get_boss_heading(game_ss)
    # DEBUG: Display boss heading arrow
    # debug.display_boss_heading(minimap_ss, boss_heading)

    # loop over options in case one poi is unreachable right now
    for i in range(len(poi_coord_to_vec)):

        # find next best room heading
        best_room_vec = pathfinding.get_next_heading(list(poi_vec_to_coord.keys()), boss_heading, i)
        # DEBUG: Display best room vector
        # debug.display_ideal_room(minimap_ss, list(poi_coord_to_vec.keys()), best_room_vec)


        # shrink map (issue with keypresses can only be so quick, smaller map = less path points returned = more accurate for key press to grid tile)
        walkable_mask_small = mask.downsample_mask(walkable_mask, block_size=MAP_SHRINK_SCALE)
        # DEBUG: display smaller map to double check resolution after shrinking
        # debug.display_mask("walkable_mask", walkable_mask)
        # debug.display_mask("downsampled_walkable_mask", debug.resize_print(walkable_mask_small, scale))

        # get shortest path to best room
        best_room_coord = poi_vec_to_coord[best_room_vec]
        adjusted_end_rc = (int(best_room_coord[1] // MAP_SHRINK_SCALE), int(best_room_coord[0] // MAP_SHRINK_SCALE))
        start_rc = (walkable_mask_small.shape[0] // 2, walkable_mask_small.shape[1] // 2)
        path, cost = pathfinding.get_shortest_path(walkable_mask_small, start_rc=start_rc, end_rc=adjusted_end_rc)

        if cost is not None:
            pathfinding.move_along_path(path, steps=12)
            break
        elif i == len(poi_coord_to_vec) - 1:
            debug.display_mask("totally stuck!!", debug.resize_print(walkable_mask_small, 5))