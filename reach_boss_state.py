from ai_state import AIState
import numpy as np
import cv2

import map_utils
import debug_prints as debug

class ReachBossState(AIState):
    def __init__(self):
        super().__init__()

        self.current_map = np.zeros((1,1))
        self.origin_offset_xy = np.array([0,0])
        self.visited_xy = []
        self.poi_pts_xy = []
        self.current_target_pt_xy = None
        self.last_known_boss_heading = None


        self.poi_proximity_radius = 15 # proximity distance two pois can be
        self.poi_visit_radius = 20 # proximity distance for a poi to be counted as visited by the player
        self.target_poi_update_distance = 30  # if no pois exist within this distance from the current poi, the best aligned one will become the next target

        self.boss_loc = None


    def update(self, ai):
        minimap_ss = ai.take_minimap_screenshot()
        game_region_ss = ai.take_game_region_screenshot()

        poi_masks = self._get_poi_masks(minimap_ss)

        # aim at nearest enemies
        enemies_mask = self._get_enemies_mask(minimap_ss)
        if np.any(enemies_mask):
            self._aim_nearest_enemy(
                enemies_mask, 
                player_loc_rc=ai.MINIMAP_CENTER_RC.copy(), 
                game_region_center_xy=ai.GAME_REGION_CENTER_XY
            )
        
        combined_poi_mask = self._combine_masks(poi_masks.values(), list(poi_masks.values())[0].shape)

        combined_poi_mask = self._fill_in_center(combined_poi_mask)

        # smooth out map with kernel (mainly used to filter out enemy outlines)
        kernel = np.ones((10, 10), np.uint8)
        combined_poi_mask = self._smooth_out_mask(combined_poi_mask, kernel)
        # DEBUG: Show final mask
        # debug.display_mask("Final Combined", combined_poi_mask)

        # update current location based on new mask
        self._parse_new_map(combined_poi_mask)

        # rooms done seperatly since looks at distance transform of all poi's instead of pixel values (can't just look at hsv value)
        poi_masks["room"] = self._get_room_mask(combined_poi_mask)
        # # DEBUG: print out all poi masks
        # for mask_name, poi_mask in poi_masks.items():
        #     debug.display_mask(mask_name, poi_mask)
        debug.display_mask("Room_mask", poi_masks["room"])

        walkable_mask, walkable_poi_mask = self._get_walkable_pois(combined_poi_mask, poi_masks, player_rc=ai.MINIMAP_CENTER_RC)
        # DEBUG: display walkable masks
        # debug.display_mask("walkable_map", walkable_mask)
        # for mask_name, poi_mask in walkable_poi_mask.items():
        #     debug.display_mask(f"{mask_name}_walkable", poi_mask)

        # check if no walkable spaces
        if np.all(walkable_mask == 0):
            self._fix_no_walkable(
                combined_poi_mask, 
                player_loc_rc=ai.MINIMAP_CENTER_RC,
                keypress_duration=ai.KEYPRESS_DURATION
            
            )
            return

        poi_pts_xy = []
        poi_relevant_masks = ["bridge/room", "room", "carpet", "ship_room"]
        for relevant_mask in poi_relevant_masks:

            poi_pts_xy.extend(self._get_mask_centers_xy(walkable_poi_mask[relevant_mask]))

        # check if we found the boss loc
        self._boss_found_check(walkable_poi_mask["carpet"])

        boss_heading_vec_xy = self._get_boss_heading_vec_xy(game_region_ss, ai.GAME_REGION_CENTER_XY)
        # DEBUG: Display boss heading arrow
        # debug.display_boss_heading(minimap_ss, boss_heading_vec_xy)

        # convert poi pts to vecs
        poi_vec_xy = []
        for pt in poi_pts_xy:
            poi_vec_xy.append(map_utils.convert_pt_to_vec(pt, center=ai.MINIMAP_CENTER_XY))
        # DEBUG: Display poi vectors
        debug.display_poi_vectors(minimap_ss, poi_vec_xy)

        # parse new pois to determine if they should be added to global pois
        for pt in poi_pts_xy:
            adjusted_x = int(pt[0] + self.origin_offset_xy[0])
            adjusted_y = int(pt[1] + self.origin_offset_xy[1])

            self._parse_new_poi((adjusted_x, adjusted_y), walkable_poi_mask["room"])

        # filter out already visited pois
        self._filter_visited_pois()

        # update the global target poi if needed
        if not any(np.array_equal(self.current_target_pt_xy, p) for p in self.poi_pts_xy):
            self._update_target_poi(boss_heading_vec_xy, player_loc_xy=ai.MINIMAP_CENTER_XY)
        
        debug.display_global_pois(self)

        # shrink map (issue with keypresses can only be so quick, smaller map = less path points returned = more accurate for key press to grid tile)
        walkable_mask_small = self._downsample_mask(walkable_mask)
        # DEBUG: display smaller map to double check resolution after shrinking
        # debug.display_mask("walkable_mask", walkable_mask)
        # debug.display_mask("downsampled_walkable_mask", debug.resize_print(walkable_mask_small, Global.MAP_SHRINK_SCALE))

        # convert the global current target poi to be with respect to the current map
        adjusted_target_pt_xy = self.current_target_pt_xy - self.origin_offset_xy
        
        # convert target poi to rc
        adjusted_target_pt_rc = map_utils.convert_pt_xy_rc(adjusted_target_pt_xy)

        path, cost = self._get_shortest_path(
            walkable_mask_small, 
            start_rc=map_utils.downscale_pt(ai.MINIMAP_CENTER_RC, self.MAP_SHRINK_SCALE), 
            end_rc=map_utils.downscale_pt(adjusted_target_pt_rc, self.MAP_SHRINK_SCALE)
        )

        if cost is not None:
            self._move_along_path(path, steps=10, keypress_duration=ai.KEYPRESS_DURATION, scale=self.MAP_SHRINK_SCALE)
        else:
            print("returned cost was None")
            input()


    def _parse_new_map(self, new_map):
        minimap_h, minimap_w = new_map.shape[:2]

        # if first new map...
        if np.all(self.current_map == 0):
            self.current_map = new_map
        else:
            padded_current_map = self._pad_map(self.current_map)
            result = cv2.matchTemplate(padded_current_map, new_map, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # safety check for low template matching confidence
            if max_val < 0.80:
                print(f"Confidence of {max_val} is bad...")
                input()

            # calculate how much we moved
            start_x, start_y = max_loc
            moved_x = start_x - minimap_w
            moved_y = start_y - minimap_h
            self.origin_offset_xy += np.array([moved_x, moved_y])

            # around the player, always prefer what we had previously
            center_h = minimap_h // 2
            center_w = minimap_w // 2
            arrow_top = 34
            arrow_bot = 18
            arrow_left = 14
            arrow_right = 14
            new_map[
                center_h - arrow_top : center_h + arrow_bot,
                center_w - arrow_left : center_w + arrow_right
            ] = self.current_map[
                center_h - arrow_top + moved_y : center_h + arrow_bot + moved_y,
                center_w - arrow_left + moved_x : center_w + arrow_right + moved_x
            ]

            # update the current map
            self.current_map = new_map

        # add current loc to visited set
        center_xy = map_utils.get_center_xy(new_map)
        self.visited_xy.append(tuple(center_xy + self.origin_offset_xy))
    
    def _get_room_mask(self, combined_poi_mask):
        # fill in small obstacles
        kernel = np.ones((20,20), np.uint8)
        map_filled = cv2.morphologyEx(combined_poi_mask, cv2.MORPH_CLOSE, kernel)

        # filter out corridors by only looking at bigger distances
        dist = cv2.distanceTransform(map_filled.astype(np.uint8), cv2.DIST_L2, 5)
        room_mask = (dist > 17).astype(np.uint8) * 255

        return room_mask

    def _parse_new_poi(self, new_poi, room_mask):

        # don't add it if it's already in the global pois
        for poi in self.poi_pts_xy:
            if poi == new_poi:
                return

        # always add the boss location as a poi
        if self.boss_loc is not None and (new_poi == self.boss_loc).all():
            self.poi_pts_xy.append(new_poi)
            return
        
        # make sure new poi isn't too close to an existing poi
        for existing_poi_xy in self.poi_pts_xy:
            distance = np.linalg.norm([existing_poi_xy[0] - new_poi[0], existing_poi_xy[1] - new_poi[1]])
            if distance < self.poi_proximity_radius:
                return

        # check if this new point falls in a room
        num_labels, labels = cv2.connectedComponents(room_mask, connectivity=4)
        local_new_poi_xy = (new_poi[0] - self.origin_offset_xy[0], new_poi[1] - self.origin_offset_xy[1])
        if labels[local_new_poi_xy[1], local_new_poi_xy[0]] != 0:

            matched_label = labels[local_new_poi_xy[1], local_new_poi_xy[0]]

            # delete existing poi's that are apart of the same connected component as this new point
            rows, cols = room_mask.shape
            for existing_poi_xy in self.poi_pts_xy.copy():
                # don't delete the boss_loc poi or the current target poi
                if (not np.array_equal(existing_poi_xy, self.boss_loc) and 
                    not np.array_equal(existing_poi_xy, self.current_target_pt_xy)
                ):
                    existing_local_poi_xy = (existing_poi_xy[0] - self.origin_offset_xy[0], existing_poi_xy[1] - self.origin_offset_xy[1])
                    if 0 <= existing_local_poi_xy[0] < cols and 0 <= existing_local_poi_xy[1] < rows:
                        if labels[existing_local_poi_xy[1], existing_local_poi_xy[0]] == matched_label:
                            self.poi_pts_xy.remove(existing_poi_xy)

            # add the center of the connected component this new point is in
            ys, xs = np.where(labels == matched_label)
            center_x = int(xs.mean())
            center_y = int(ys.mean())
            self.poi_pts_xy.append(tuple(np.array([center_x, center_y]) + self.origin_offset_xy))


        # condensing down poi's to the center is only done to rooms for now
        # if the point doesn't lie in a room from the room mask, add it to potential pois just to be safe
        else:
            self.poi_pts_xy.append(new_poi)
    
    def _boss_found_check(self, boss_room_mask):

        # only check for boss loc if not already found
        if self.boss_loc is None:
            num_labels, labels = cv2.connectedComponents(boss_room_mask, connectivity=4)

            for label in range(1, num_labels):

                mask = (labels == label).astype(np.uint8)
                M = cv2.moments(mask)

                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])

                    self.boss_loc = np.array([center_x, center_y]) + self.origin_offset_xy

    def _filter_visited_pois(self):
        global_pois_copy = self.poi_pts_xy.copy()
        for poi_xy in global_pois_copy:
            for visited_xy in self.visited_xy:
                distance = np.linalg.norm([poi_xy[0] - visited_xy[0], poi_xy[1] - visited_xy[1]])
                if distance < self.poi_visit_radius:

                    # if boss loc is about to be counted as visited
                    if self.boss_loc is not None and (poi_xy == self.boss_loc).all():
                        self.state_done = True
                        print("----------------Marking boss as visited, moving on to defeating boss-----------------")

                    self.poi_pts_xy.remove(poi_xy)
                    break
    
    def _update_target_poi(self, boss_heading_vec_xy, player_loc_xy):
            center_xy = player_loc_xy + self.origin_offset_xy

            # check if there's a nearby poi
            for xy in self.poi_pts_xy:
                if np.linalg.norm(xy - center_xy) < self.target_poi_update_distance:
                    print(f"Found nearby poi of {xy}. Aiming for this instead of best aligned")
                    self.current_target_pt_xy = xy
                    return

            # turn global poi pts into vecs
            pois_vec_xy = []
            for xy in self.poi_pts_xy:
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
            if len(self.poi_pts_xy) == 0:
                print("No pois found, running towards boss")
                return boss_heading_vec_xy

            dots = [np.dot(r/np.linalg.norm(r), boss_heading_vec_xy) for r in pois_vec_xy]
            best_index = np.argmax(dots)
            best_vec_xy = pois_vec_xy[best_index]
            best_pt_xy = best_vec_xy + center_xy

            self.current_target_pt_xy = best_pt_xy

    def _get_boss_heading_vec_xy(self, game_ss, game_region_center_xy):
        template = cv2.imread('sprites/boss_icon.png', cv2.IMREAD_GRAYSCALE)

        game_bgr = np.array(game_ss)[:,:,:3].copy()  # MSS gives a 4th alpha channel, so shrink this down to 3 channels
        game_gray = cv2.cvtColor(game_bgr, cv2.COLOR_BGR2GRAY)  # convert the 3 channels to grayscale

        result = cv2.matchTemplate(game_gray, template, cv2.TM_CCOEFF_NORMED)  

        _, max_val, _, max_loc_xy = cv2.minMaxLoc(result)

        if max_val < 0.5:
            print(f"Bad confidence, assuming boss icon dissapeared, using last known heading")
            return self.last_known_boss_heading

        template_height, template_width = template.shape
        template_mid = np.array([max_loc_xy[0] + template_width // 2, max_loc_xy[1] + template_height // 2])

        boss_heading_vec = template_mid - game_region_center_xy
        self.last_known_boss_heading = boss_heading_vec

        return boss_heading_vec

    def _pad_map(self, map):
        height, width = map.shape[:2]

        padded_map = np.zeros((3*height, 3*width), dtype=map.dtype)
        padded_map[height:height*2, width:width*2] = map

        return padded_map
