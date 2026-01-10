import numpy as np

class Global:
    poi_pts_xy = []
    origin_offset_xy = np.array([0,0])
    current_map = np.zeros((1,1))
    visited_xy = []
    current_target_pt_xy = None

    MAP_SHRINK_SCALE = 2
    KEYPRESS_DURATION = 0
    PLAYER_OFFSET_XY = np.array([260, 266])

    @classmethod
    def get_map_dim_xy(cls):
        return (cls.map.shape[1], cls.map.shape[0])
    
    @classmethod
    def get_map_dim_rc(cls):
        return (cls.map.shape[0], cls.map.shape[1])

    @classmethod
    def update_target_poi(cls, boss_heading_vec_xy, closest_poi_distance):
        center_xy = Global.PLAYER_OFFSET_XY + Global.origin_offset_xy

        # check if there's a nearby poi
        for xy in Global.poi_pts_xy:
            if np.linalg.norm(xy - center_xy) < closest_poi_distance:
                print(f"Found nearby poi of {xy}. Aiming for this instead of best aligned")
                Global.current_target_pt_xy = xy
                return

        # turn global poi pts into vecs
        pois_vec_xy = []
        for xy in Global.poi_pts_xy:
            pois_vec_xy.append(xy - center_xy)

        # img = np.zeros((600, 600, 3), dtype=np.uint8)
        # for v in pois_vec_xy:
        #     end = (center_xy + v).astype(int)

        #     cv2.arrowedLine(
        #         img,
        #         center_xy,
        #         tuple(end),
        #         color=(0, 255, 0),   # green
        #         thickness=2,
        #         tipLength=0.2
        #     )
        # cv2.arrowedLine(
        #     img,
        #     center_xy,
        #     tuple(center_xy + boss_heading_vec_xy),
        #     color=(255,0,0),
        #     thickness=2,
        #     tipLength=0.2
        # )

        # cv2.imshow("POI Vectors", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # If no pois, run towards boss
        if len(Global.poi_pts_xy) == 0:
            print("No pois found, running towards boss")
            return boss_heading_vec_xy

        dots = [np.dot(r/np.linalg.norm(r), boss_heading_vec_xy) for r in pois_vec_xy]
        best_index = np.argmax(dots)
        best_vec_xy = pois_vec_xy[best_index]
        best_pt_xy = best_vec_xy + center_xy

        Global.current_target_pt_xy = best_pt_xy