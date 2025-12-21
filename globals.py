import numpy as np

class Global:
    poi_pts_xy = set()
    current_loc_xy = np.array([0, 0])
    previous_map = np.zeros((1,1))

    MAP_SHRINK_SCALE = 2
    KEYPRESS_DURATION = 0

    @classmethod
    def update_previous_map(cls, new_map):
        Global.previous_map = new_map
        height, width = Global.previous_map.shape[:2]

        new_map = np.zeros((3*height, 3*width), dtype=Global.previous_map.dtype)
        new_map[height:height*2, width:width*2] = Global.previous_map
        Global.previous_map = new_map

    @classmethod
    def get_map_dim_xy(cls):
        return (cls.map.shape[1], cls.map.shape[0])
    
    @classmethod
    def get_map_dim_rc(cls):
        return (cls.map.shape[0], cls.map.shape[1])