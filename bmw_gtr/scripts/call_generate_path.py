#!/usr/bin/python3
import cv2 as cv
from path_planning4_mod import PathPlanning  # adjust import if filename differs
import numpy as np

if __name__ == "__main__":
    # Load your map image
    track = cv.imread('data/final_map.png')

    # Initialize planner
    planner = PathPlanning(track)

    # Define a list of nodes you want the path to pass through
    nodes_to_visit = [111, 120]  # example sequence

    # Generate the path
    planner.generate_path_passing_through(nodes_to_visit, step_length=0.1)

    print(planner.path_data)

    # Recompute yaw for every interpolated step
    path_with_yaw = []
    for i in range(len(planner.path)-1):
        x, y = planner.path[i]
        x_next, y_next = planner.path[i+1]
        yaw = np.rad2deg(np.arctan2(y_next - y, x_next - x))
        path_with_yaw.append((x, y, yaw))

    # Add last point with previous yaw
    path_with_yaw.append((*planner.path[-1], yaw))

    print(path_with_yaw)

    # Draw the path
    planner.draw_path()

    # Scale down the image before displaying
    scale = 0.25  # 50% of original size
    resized_map = cv.resize(planner.map, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

    # Show the scaled image
    cv.imshow("Generated Path", resized_map)
    cv.waitKey(0)
    cv.destroyAllWindows()
