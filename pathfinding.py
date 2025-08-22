import heapq
import numpy as np

def direction_guided_search(grid, start, boss_heading, room_headings, max_steps=500):
    """
    grid: 2D numpy array, 0 = walkable, 1 = obstacle
    start: (row, col)
    __heading: (dx, dy) preferred direction
    max_steps: cutoff to avoid infinite loops
    """


    rows, cols = grid.shape
    visited = set()
    came_from = {}

    # priority queue entries: (priority, (r, c))
    frontier = []
    heapq.heappush(frontier, (0, start))

    steps = 0
    while frontier and steps <= max_steps:
        _, (r, c) = heapq.heappop(frontier)
        steps += 1

        if (r, c) in visited:
            continue
        visited.add((r, c))

        # STOP CONDITION
        if steps > 200:
            # reconstruct path
            path = []
            cur = (r, c)
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            return path

        # explore neighbors
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1:
                if (nr, nc) not in visited:
                    came_from[(nr, nc)] = (r, c)

                    # bias = alignment with heading
                    vec_raw = np.array([nr - start[0], nc - start[1]])
                    vec_norm = vec_raw / np.linalg.norm(vec_raw)  # normalize vec so dot product doesn't get influenced by direction
                    boss_dot = np.dot(vec_norm, boss_heading)  # bigger = better alignment

                    # get dot product for each room headings
                    room_dots = []
                    for room in room_headings:
                        room_dots.append(np.dot(vec_norm, room))

                    dist = np.linalg.norm(vec_raw)
                    priority = dist - 0.1*boss_dot - max(room_dots)  # weight toward heading
                    heapq.heappush(frontier, (priority, (nr, nc)))
    
    return None  # no path found

def map_delta_to_key(dr, dc):
    if dr == -1 and dc == 0:
        return 'w'
    elif dr == 1 and dc == 0:
        return 's' 
    elif dr == 0 and dc == -1:
        return 'a'  
    elif dr == 0 and dc == 1:
        return 'd'
    else:
        return None  # no movement