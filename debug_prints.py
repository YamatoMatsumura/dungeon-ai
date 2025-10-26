import cv2
import matplotlib.pyplot as plt
import numpy as np

import color_mask as mask


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

def display_poi_vectors(minimap_ss, poi_vec_to_coord):
    # get player xy
    height, width = minimap_ss.shape[:2]
    player_x = width // 2
    player_y = height // 2

    minimap_ss_copy = minimap_ss.copy()

    # draw each poi vector
    for x, y in poi_vec_to_coord.keys():
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
    return cv2.resize(img, (img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

def display_shorest_path(walkable_tiles_small, centroids, scale, indices):
    plt.imshow(walkable_tiles_small, cmap="gray")
    for cx, cy in centroids:
        plt.plot(cx // scale, cy // scale, 'ro')

    y, x = zip(*indices)
    plt.plot(x, y, 'b-')
    plt.show()