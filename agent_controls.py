import mss
import time
import numpy as np
import cv2
from scipy import ndimage
import pyautogui

from pathfinding import get_nearest_target_rc
from key_press import press_keys, VK_CODES
from globals import Global
import map_utils
import pathfinding
import debug_prints as debug
import color_mask as mask

def initialize_pixels_per_step():
    lower_bound = None
    upper_bound = None

    keypress_duration = 0.001  # assumed to safely return a pixel offset of 0
    while lower_bound is None or upper_bound is None:
        with mss.mss() as sct:

            # get minimap screenshot
            minimap_region = {"top": 5, "left": 2032, "width": 522, "height": 533}
            initial_ss = np.array(sct.grab(minimap_region))

            # pad initial screenshot so template matching works
            pad = 20
            expanded_initial = cv2.copyMakeBorder(
                initial_ss,
                pad, pad, pad, pad,
                borderType=cv2.BORDER_REPLICATE
            )

            time.sleep(0.01)
            press_keys([VK_CODES["w"]], duration=keypress_duration)
            time.sleep(0.01)
            moved_ss = np.array(sct.grab(minimap_region))

            result = cv2.matchTemplate(expanded_initial, moved_ss, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            time.sleep(0.01)
            press_keys([VK_CODES["s"]], duration=keypress_duration)
            time.sleep(0.01)
            pixels_per_step = pad - max_loc[1]

            if pixels_per_step == 1 and lower_bound is None:
                lower_bound = keypress_duration
            elif pixels_per_step > 1 and upper_bound is None:
                upper_bound = keypress_duration
            else:
                keypress_duration += 0.001
            
            time.sleep(0.01)

    Global.MIN_KEYPRESS_DURATION = (lower_bound + 0.25*(upper_bound + lower_bound))  # bias towards lower end of bound
    print(f"Found min keypress was {Global.MIN_KEYPRESS_DURATION}")

def check_min_duration():
    with mss.mss() as sct:
        # get minimap screenshot
        minimap_region = {"top": 5, "left": 2032, "width": 522, "height": 533}
        initial_ss = np.array(sct.grab(minimap_region))

        # pad initial screenshot so template matching works
        pad = 20
        expanded_initial = cv2.copyMakeBorder(
            initial_ss,
            pad, pad, pad, pad,
            borderType=cv2.BORDER_REPLICATE
        )

        time.sleep(0.01)
        press_keys([VK_CODES["w"]], duration=Global.MIN_KEYPRESS_DURATION*2)
        time.sleep(0.01)
        moved_ss = np.array(sct.grab(minimap_region))

        result = cv2.matchTemplate(expanded_initial, moved_ss, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        time.sleep(0.01)
        press_keys([VK_CODES["s"]], duration=Global.MIN_KEYPRESS_DURATION*2)
        time.sleep(0.01)
        pixels_per_step = pad - max_loc[1]

        print(f"pixels per step: {pixels_per_step}")

def map_delta_to_key(dr, dc):
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

def move_along_path(path, steps, scale=1, slower_movement_adjustment=1):
    # Make sure only going at max len(path)
    if steps > len(path):
        steps = len(path)
    
    key_counts = []
    last_key = None
    count = 0
    for i in range(1, steps):
        dr = path[i][0] - path[i-1][0]
        dc = path[i][1] - path[i-1][1]

        key = map_delta_to_key(dr, dc)
        
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
        press_keys(keys_to_press, duration=Global.MIN_KEYPRESS_DURATION*count*scale*slower_movement_adjustment)

def press_f():
    press_keys([VK_CODES['f']])

def aim_nearest_enemy(enemies_mask, player_loc_rc, game_center_xy):
    # manual player loc adjust since weapon fires a little bit above center
    player_loc_rc[0] += 5

    nearest_enemy_pt_rc = get_nearest_target_rc(enemies_mask, player_loc_rc)
    nearest_enemy_pt_xy = map_utils.convert_pt_xy_rc(nearest_enemy_pt_rc)
    player_loc_xy = map_utils.convert_pt_xy_rc(player_loc_rc)
    nearest_enemy_vec_xy = map_utils.convert_pt_to_vec(nearest_enemy_pt_xy, player_loc_xy)

    direction = nearest_enemy_vec_xy / np.linalg.norm(nearest_enemy_vec_xy)
    aim_distance = 300

    aim_loc_xy = game_center_xy + direction * aim_distance
    pyautogui.moveTo(aim_loc_xy[0], aim_loc_xy[1])

    return

def shadow_nearest_enemy(enemies_mask, player_loc_rc, map, move_distance):
    nearest_enemy_pt_rc = get_nearest_target_rc(enemies_mask, player_loc_rc)

    path, cost = pathfinding.get_shortest_path(
        mask.downsample_mask(map),
        start_rc=map_utils.downscale_pt(player_loc_rc), 
        end_rc=map_utils.downscale_pt(nearest_enemy_pt_rc)
    )
    # debug.display_pathfinding(mask.downsample_mask(map), path, map_utils.downscale_pt(player_loc_rc), map_utils.downscale_pt(nearest_enemy_pt_rc))

    move_along_path(path, steps=move_distance)
