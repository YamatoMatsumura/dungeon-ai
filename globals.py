import numpy as np

class Global:
    map = np.zeros((1000, 1000), dtype=np.uint8)
    poi_pts_xy = set()
    MAP_SHRINK_SCALE = 2
    KEYPRESS_DURATION = 0

    @classmethod
    def get_map_dim_xy(cls):
        return (cls.map.shape[1], cls.map.shape[0])
    
    @classmethod
    def get_map_dim_rc(cls):
        return (cls.map.shape[0], cls.map.shape[1])