import mss
import numpy as np
import cv2
import pydirectinput
import matplotlib.pyplot as plt

import time  # to delay screenshot

import color_mask as mask
import pathfinding

time.sleep(2)
while True:

    # grab screenshots
    with mss.mss() as sct:
        # minimap screenshot
        minimap_region = {"top": 0, "left": 2027, "width": 531, "height": 545}
        minimap_ss = np.array(sct.grab(minimap_region))

        # boss direction screenshot
        boss_region = {"top": 0, "left": 0, "width": 2025, "height": 1600}
        boss_ss = np.array(sct.grab(boss_region))

    # # DEBUG: Get HSV values for stuff on minimap
    # map = cv2.cvtColor(minimap_ss, cv2.COLOR_BGR2HSV)
    # plt.imshow(map)
    # plt.show()

    # get boolean grid for walkable spaces
    walkable_grid = mask.mask_minimap(minimap_ss)
    
    # get heading for boss
    boss_heading = mask.get_boss_heading(boss_ss)

    # get headings for each room (helps push bot take paths through rooms and not greedy dash to boss)
    room_headings = mask.get_room_headings(minimap_ss)

    # shrink map (issue with keypresses can only be so quick, smaller map = less path points returned = more accurate for key press to grid tile)
    shrink_size = 5
    walkable_grid_small = mask.downsample_mask(walkable_grid, shrink_size)

    # # DEBUG: display smaller map to double check resolution after shrinking
    # img = (walkable_grid_small.astype(np.uint8)) * 255
    # cv2.imshow("map", mask.resize_print(img, shrink_size))
    # cv2.waitKey(0)

    # get desired path
    start_row = walkable_grid_small.shape[0] // 2
    start_col = walkable_grid_small.shape[1] // 2
    path = pathfinding.direction_guided_search(
        walkable_grid_small, 
        start=(start_row, start_col), 
        boss_heading=boss_heading, 
        room_headings=room_headings
    )


    # DEBUG: overlays path over minimap
    grid_color = np.stack([walkable_grid_small*255]*3, axis=-1).astype(np.uint8)  # grayscale to BGR
    for r, c in path:
        grid_color[r, c] = [0, 0, 255] 

    # cv2.imshow("Path", mask.resize_print(grid_color, shrink_size))
    # cv2.waitKey(0)


    # translate coordinate path into keypresses
    keys = []
    for i in range(len(path) - 1):
        current = path[i]
        next = path[i+1]

        delta_x = next[0] - current[0]
        delta_y = next[1] - current[1]

        key = pathfinding.map_delta_to_key(delta_x, delta_y)
        keys.append(key)

        if key:
            pydirectinput.keyDown(key)
            time.sleep(0.001)
            pydirectinput.keyUp(key)