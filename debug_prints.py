import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.graph import route_through_array


import color_mask as mask
from globals import Global

def display_BGR(ss):
    cv2.imshow("minimap", cv2.cvtColor(ss, cv2.COLOR_BGRA2BGR))
    cv2.waitKey(0)

def display_HSV(ss):
    map = cv2.cvtColor(ss, cv2.COLOR_BGR2HSV)
    plt.imshow(map)
    plt.show()

def display_mask(window_name, boolean_array):
    cv2.imshow(window_name, boolean_array)
    cv2.waitKey(0)

def display_poi_vectors(minimap_ss, poi_vec_xy):
    # get player xy
    height, width = minimap_ss.shape[:2]
    player_x = width // 2
    player_y = height // 2

    minimap_ss_copy = minimap_ss.copy()

    # draw each poi vector
    for x, y in poi_vec_xy:
        end_point = (int(player_x + x), int(player_y + y))

        cv2.arrowedLine(
            minimap_ss_copy,
            (player_x, player_y),
            end_point,
            color=(0,255,0),
            thickness=2,
            tipLength=0.2
        )

    cv2.imshow("POI Vectors", minimap_ss_copy)
    cv2.waitKey(0)

def display_boss_heading(minimap_ss, boss_heading):
   # display boss heading
    height, width = minimap_ss.shape[:2]
    player = [width // 2, height // 2]

    boss_heading_test = minimap_ss.copy()

    end_point = (int(player[0] + boss_heading[0]*0.2), int(player[1] + boss_heading[1]*0.2))  # scaled since based on game ss and not minimap ss

    cv2.arrowedLine(
        boss_heading_test,
        player,
        end_point,
        color=(0,255,0),
        thickness=2,
        tipLength=0.2
    )

    cv2.imshow("Boss Heading Test", boss_heading_test)
    cv2.waitKey(0)

def display_ideal_room(minimap_ss, centroids, best_room_vec):
    # Visualize all rooms
    best_room_vec_test = minimap_ss.copy()
    for cx, cy in centroids:
        cv2.circle(best_room_vec_test, (int(cx), int(cy)), radius=5, color=(0,0,255), thickness=-1)  # red dots

    # draw ideal room arrow
    height, width = minimap_ss.shape[:2]
    player = (height // 2, width // 2)
    ideal_room = (int(player[0] + best_room_vec[0]), int(player[1] + best_room_vec[1]))
    cv2.arrowedLine(
        best_room_vec_test,
        player,
        ideal_room,
        color=(0,255,0),
        thickness=2,
        tipLength=0.2
    )

    cv2.imshow("Best Room Vector Test", best_room_vec_test)
    cv2.waitKey(0)

def resize_print(img, scale):
    return cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_NEAREST)

def display_shorest_path(walkable_tiles_small, centroids, scale, indices):
    plt.imshow(walkable_tiles_small, cmap="gray")
    for cx, cy in centroids:
        plt.plot(cx // scale, cy // scale, 'ro')

    y, x = zip(*indices)
    plt.plot(x, y, 'b-')
    plt.show()

def display_pathfinding(walkable_mask_small, path_indices, player_rc, end_rc):
    debug_map = walkable_mask_small.copy()
    debug_map_color = cv2.cvtColor(debug_map, cv2.COLOR_GRAY2BGR)

    if path_indices is not None:
        path_color = (255, 255, 0)
        radius = 1
        
        for r, c in path_indices:
            cv2.circle(debug_map_color, (c, r), radius, path_color, -1)

    # Draw start (green) and end (red)
    cv2.circle(debug_map_color, (player_rc[1], player_rc[0]), 3, (0, 255, 0), -1)
    cv2.circle(debug_map_color, (end_rc[1], end_rc[0]), 3, (0, 0, 255), -1)

    # Show result
    cv2.imshow("path", resize_print(debug_map_color, 2))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_global_pois(POI_VISIT_RADIUS, TARGET_POI_UPDATE_DISTANCE):
    pois = np.array(list(Global.poi_pts_xy), dtype=int)
    visited = np.array(list(Global.visited_xy), dtype=int)

    # Include origin and manual point in bounds
    extra_points = np.array([
        [0, 0],
        [261, 266],
    ], dtype=int)

    if pois.size == 0:
        all_points = extra_points
    else:
        all_points = np.vstack([pois, extra_points])

    # Padding for labels
    padding = 40

    # Find extents relative to origin
    min_x = all_points[:, 0].min()
    max_x = all_points[:, 0].max()
    min_y = all_points[:, 1].min()
    max_y = all_points[:, 1].max()

    # Image size
    w = (max_x - min_x) + padding * 2
    h = (max_y - min_y) + padding * 2

    # Origin position in image
    origin_x = -min_x + padding
    origin_y = -min_y + padding

    img = np.ones((h, w, 3), dtype=np.uint8) * 255

    # Draw axes (optional but very helpful)
    cv2.line(img, (0, origin_y), (w, origin_y), (200, 200, 200), 1)
    cv2.line(img, (origin_x, 0), (origin_x, h), (200, 200, 200), 1)

    # Draw POIs
    for x, y in pois:
        px = x + origin_x
        py = y + origin_y

        if Global.current_target_pt_xy[0] == x and Global.current_target_pt_xy[1] == y:
            cv2.circle(img, (px, py), 4, (0, 128, 0), -1)
        else:
            cv2.circle(img, (px, py), 4, (0, 0, 255), -1)
        cv2.putText(
            img,
            f"({x}, {y})",
            (px + 5, py + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
    
    # Draw visited locs
    for x, y in visited:
        px = x + origin_x
        py = y + origin_y
        cv2.circle(img, (px, py), POI_VISIT_RADIUS, (60, 28, 100), 2)

    # Draw origin
    cv2.circle(img, (origin_x, origin_y), 6, (255, 0, 0), -1)
    cv2.putText(
        img,
        "(0,0)",
        (origin_x + 5, origin_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 0, 0),
        1
    )

    # Draw manual point (261, 266)
    mx, my = 261, 266
    cv2.circle(
        img,
        (mx + origin_x, my + origin_y),
        6,
        (0, 255, 0),
        -1
    )

    # draw current location
    offset_x, offset_y = Global.origin_offset_xy
    cv2.circle(img, (offset_x + mx + origin_x, offset_y + my + origin_y), 6, (255, 0, 0), -1)
    cv2.circle(img, (offset_x + mx + origin_x, offset_y + my + origin_y), TARGET_POI_UPDATE_DISTANCE, (255, 0, 255), 2)  # for nearby poi finding
    cv2.putText(
        img,
        f"({offset_x}, {offset_y})",
        (offset_x + mx + origin_x + 5, offset_y + my + origin_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 0, 0),
        1
    )

    cv2.imshow("POIs (Origin Centered)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()