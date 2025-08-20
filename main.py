import mss
import numpy as np
import cv2
import time


time.sleep(2)
with mss.mss() as sct:
    # define region for minimap
    region = {"top": 0, "left": 2027, "width": 531, "height": 545}
    screenshot = np.array(sct.grab(region))
    
    cv2.imshow("minimap", cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR))
    cv2.waitKey(0)
