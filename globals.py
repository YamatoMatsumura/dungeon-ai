import numpy as np

class Global:
    poi_pts_xy = set()
    origin_offset_xy = np.array([0,0])
    current_map = np.zeros((1,1))

    MAP_SHRINK_SCALE = 2
    KEYPRESS_DURATION = 0

    @classmethod
    def get_map_dim_xy(cls):
        return (cls.map.shape[1], cls.map.shape[0])
    
    @classmethod
    def get_map_dim_rc(cls):
        return (cls.map.shape[0], cls.map.shape[1])