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
        # # Debug: Getting minimap pixel region
        # debug.display_BGR(minimap_ss)

        # # DEBUG: Get HSV values for stuff on minimap
        # debug.display_HSV(minimap_ss)

    walkable_tiles = mask.get_walkable_tiles(minimap_ss)
    centroids = pathfinding.get_room_centroids(walkable_tiles, minimap_ss)

    # convert centroids to vectors
    room_coord_to_vec, room_vec_to_coord = pathfinding.get_room_vectors(centroids, minimap_ss)
    # # DEBUG: Display Room Vectors
    # debug.display_room_vectors(minimap_ss, room_vec_to_coord)

    boss_heading = pathfinding.get_boss_heading(game_ss)
    # # DEBUG: Display boss heading arrow
    # debug.display_boss_heading(minimap_ss, boss_heading)

    # find which room will get us to boss
    best_room_vec = pathfinding.get_best_room_heading(list(room_vec_to_coord.keys()), boss_heading)
    # # DEBUG: Display best room vector
    # debug.display_ideal_room(minimap_ss, centroids, best_room_vec)

    # shrink map (issue with keypresses can only be so quick, smaller map = less path points returned = more accurate for key press to grid tile)
    scale = 5
    walkable_tiles_small = mask.downsample_mask(walkable_tiles, block_size=scale)
    # # DEBUG: display smaller map to double check resolution after shrinking
    # debug.display_downsample_diff(walkable_tiles, walkable_tiles_small, scale)

    # get shortest path to best room
    path = pathfinding.get_shortest_path(walkable_tiles_small, minimap_ss, scale, room_vec_to_coord, best_room_vec)
    # # DEBUG: display map overlayed with shortest path
    # debug.display_shorest_path(walkable_tiles_small, centroids, scale, path)

    pathfinding.move_along_path(minimap_ss, scale, path)
