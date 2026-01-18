import mss
import numpy as np
import time
import cv2

from key_press import press_keys, VK_CODES
from reach_boss_state import ReachBossState
import map_utils


class DungeonAI:

    def __init__(self):

        # minimap dimensons for screenshot
        self.minimap_top = 5
        self.minimap_left = 2032
        self.minimap_width = 522
        self.minimap_height = 533

        # game region dimensions for screenshot
        self.game_region_top = 0
        self.game_region_left = 0
        self.game_region_width = 2025
        self.game_region_height = 1600

        self.KEYPRESS_DURATION = None
        self.MINIMAP_CENTER_RC = None
        self.GAME_REGION_CENTER_XY = None

        self.running = True
        self.current_state = None

    def start(self):
        self.initialize()

        self.current_state = ReachBossState()
        while self.running:
            self.current_state.update(self)
                
    
    def initialize(self):
        self._initialize_keypress_duration()
        self._check_keypress_duration_accuracy()

        minimap_ss = self.take_minimap_screenshot()
        self.MINIMAP_CENTER_RC = map_utils.get_center_rc(minimap_ss)
        game_region_ss = self.take_game_region_screenshot()
        self.GAME_REGION_CENTER_XY = map_utils.get_center_xy(game_region_ss)

    def take_minimap_screenshot(self):
        with mss.mss() as sct:
            minimap_region = {
                "top": self.minimap_top, 
                "left": self.minimap_left, 
                "width": self.minimap_width, 
                "height": self.minimap_height
            }
            minimap_ss = np.array(sct.grab(minimap_region))

        return minimap_ss

    def take_game_region_screenshot(self):
        with mss.mss() as sct:
            game_region = {
                "top": self.game_region_top, 
                "left": self.game_region_left, 
                "width": self.game_region_width, 
                "height": self.game_region_height
            }
            game_ss = np.array(sct.grab(game_region))
        
        return game_ss

    def _initialize_keypress_duration(self):
        lower_bound = None
        upper_bound = None

        keypress_duration = 0.001  # assumed to safely return a pixel offset of 0
        while lower_bound is None or upper_bound is None:
            initial_ss = self.take_minimap_screenshot()

            # pad initial screenshot so template matching works
            pad = 20
            expanded_initial = cv2.copyMakeBorder(
                initial_ss,
                pad, pad, pad, pad,
                borderType=cv2.BORDER_REPLICATE
            )

            # move up
            time.sleep(0.01)
            press_keys([VK_CODES["w"]], duration=keypress_duration)
            time.sleep(0.01)

            # take screenshot to determine how much we moved
            moved_ss = self.take_minimap_screenshot()

            result = cv2.matchTemplate(expanded_initial, moved_ss, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # move back down to reset position
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

        self.KEYPRESS_DURATION = (lower_bound + 0.25*(upper_bound + lower_bound))  # bias towards lower end of bound
    
    def _check_keypress_duration_accuracy(self):
        initial_ss = self.take_minimap_screenshot()

        # pad initial screenshot so template matching works
        pad = 20
        expanded_initial = cv2.copyMakeBorder(
            initial_ss,
            pad, pad, pad, pad,
            borderType=cv2.BORDER_REPLICATE
        )

        # move up for twice the duration
        time.sleep(0.01)
        press_keys([VK_CODES["w"]], duration=self.KEYPRESS_DURATION)
        time.sleep(0.01)
        press_keys([VK_CODES["w"]], duration=self.KEYPRESS_DURATION)

        moved_ss = self.take_minimap_screenshot()

        result = cv2.matchTemplate(expanded_initial, moved_ss, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # reset back to starting pos
        press_keys([VK_CODES["s"]], duration=self.KEYPRESS_DURATION)
        press_keys([VK_CODES["s"]], duration=self.KEYPRESS_DURATION)
        pixels_per_step = pad - max_loc[1]

        print(f"[Keypress Check] Duration: {self.KEYPRESS_DURATION:.4f}s | "
            f"Pixels moved per step: {pixels_per_step} (should be close to 2)")