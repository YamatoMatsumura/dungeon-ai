import mss
import numpy as np

import time  # to delay screenshot

import color_mask as mask
import pathfinding
import debug_prints as debug

time.sleep(2)
# while True:
while True:

    with mss.mss() as sct:

        # get minimap screenshot
        minimap_region = {"top": 0, "left": 2027, "width": 531, "height": 545}
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
    # # DEBUG: Display boss heading arrow
    # debug.display_boss_heading(minimap_ss, boss_heading)

    # find which room will get us to boss
    best_room_vec = pathfinding.get_best_room_heading(list(poi_vec_to_coord.keys()), boss_heading)
    # DEBUG: Display best room vector
    # debug.display_ideal_room(minimap_ss, list(poi_coord_to_vec.keys()), best_room_vec)

    eroded_walkable_mask = pathfinding.shrink_walkable_mask(walkable_mask)
    # DEBUG: Display before and after erode
    # debug.display_mask("Before erode", walkable_mask)
    # debug.display_mask("After Erode", eroded_walkable_mask)

    # shrink map (issue with keypresses can only be so quick, smaller map = less path points returned = more accurate for key press to grid tile)
    scale = 5
    walkable_mask_small = mask.downsample_mask(walkable_mask, block_size=scale)
    # DEBUG: display smaller map to double check resolution after shrinking
    # debug.display_mask("walkable_mask", walkable_mask)
    # debug.display_mask("downsampled_walkable_mask", debug.resize_print(walkable_mask_small, scale))

    # try:
    #     # get shortest path to best room
    #     path = pathfinding.get_shortest_path(walkable_mask_small, minimap_ss, scale, poi_vec_to_coord, best_room_vec)
    #     # DEBUG: display map overlayed with shortest path
    #     debug.display_shorest_path(walkable_mask_small, list(poi_coord_to_vec.keys()), scale, path)

    #     pathfinding.move_along_path(minimap_ss, scale, path)
    # except Exception as e:
    #     print(f"Error: {e}")

    # get shortest path to best room
    path, cost = pathfinding.get_shortest_path(walkable_mask_small, minimap_ss, scale, poi_vec_to_coord, best_room_vec)

    if cost is None:
        debug.display_mask("stuck!!", walkable_mask)
        print("No path Found...retrying...")
    else:
        # DEBUG: display map overlayed with shortest path
        # debug.display_shorest_path(walkable_mask_small, list(poi_coord_to_vec.keys()), scale, path)
        pathfinding.move_along_path(minimap_ss, scale, path, steps=15)