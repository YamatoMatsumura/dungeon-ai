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

def display_room_vectors(minimap_ss, room_vec_to_coord):
    # display vectors
    height, width = minimap_ss.shape[:2]
    player = [width // 2, height // 2]

    room_vec_test = minimap_ss.copy()

    for vec in room_vec_to_coord.keys():
        end_point = (int(player[0] + vec[0]), int(player[1] + vec[1]))  # adds vector to current player pos to turn into x,y coord

        cv2.arrowedLine(
            room_vec_test,
            player,
            end_point,
            color=(0,255,0),
            thickness=2,
            tipLength=0.2
        )

    cv2.imshow("Room Vectors Test", room_vec_test)
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

def display_downsample_diff(walkable_tiles, walkable_tiles_small, scale):
    cv2.imshow("map", walkable_tiles)
    cv2.waitKey(0)
    img = (walkable_tiles_small.astype(np.uint8)) * 255
    cv2.imshow("map", mask.resize_print(img, scale))
    cv2.waitKey(0)

def display_shorest_path(walkable_tiles_small, centroids, scale, indices):
    plt.imshow(walkable_tiles_small, cmap="gray")
    for cx, cy in centroids:
        plt.plot(cx // scale, cy // scale, 'ro')

    y, x = zip(*indices)
    plt.plot(x, y, 'b-')
    plt.show()
    print(len(indices))