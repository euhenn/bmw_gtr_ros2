#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from path_planning_eugen import PathPlanning
from reference_generation_velocity import TrajectoryGeneration

# === Load log ===
if len(sys.argv) < 2:
    print("Usage: python3 plot.py logs/mpc_log_<timestamp>.csv")
    sys.exit(1)

log_path = sys.argv[1]
data = pd.read_csv(log_path)

# Convert columns to numeric
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# === Extract signals ===
x = data["x"].to_numpy()
y = data["y"].to_numpy()
t = data["t"].to_numpy()
v = data["v"].to_numpy()
v_cmd = data["v_cmd"].to_numpy()
delta_actual = data["delta_actual"].to_numpy()
delta_cmd = np.rad2deg(data["delta_cmd"].to_numpy())
e_y = data["e_y"].to_numpy()
e_psi = data["e_psi"].to_numpy()
a_cmd = data["a_cmd"].to_numpy()




nodes_to_visit = [73, 97, 125, 150]
traj_gen = TrajectoryGeneration(ds=0.1, N_horizon=50, v_max=0.5, v_min=0.3, use_curvature_velocity=True, smooth_velocity=True)
# Draw points on path
traj_gen.planner.draw_path_nodes(traj_gen.planner.route_list)
# Draw whole path
full_path = np.column_stack((traj_gen.planner.x_ref, traj_gen.planner.y_ref))
traj_gen.planner.draw_path(full_path)

#path by car
path_xy = np.column_stack((x, y))
traj_gen.planner.draw_path(np.array([path_xy], dtype=np.float32), color=(0, 255, 255), thickness=3)

# Show map
traj_gen.planner.show_map_resized(roi_height_ratio=1, roi_width_ratio=1, scale=0.3)
cv.waitKey(0)




# === --- PLOT 1: Errors (lateral + heading) --- ===
plt.figure(figsize=(10, 5))
plt.plot(t, e_y, label="Lateral Error $e_y$ [m]", color="#0077b6", linewidth=2)
plt.plot(t, e_psi, label="Heading Error $e_{ψ}$ [rad]", color="#ef476f", linewidth=2)
plt.title("Tracking Errors", fontsize=14, fontweight='bold')
plt.xlabel("Time [s]", fontsize=12)
plt.ylabel("Error", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# === --- PLOT 2: Actual vs Commanded Speed --- ===
plt.figure(figsize=(10, 5))
plt.plot(t, v_cmd, "--", label="Commanded Speed [m/s]", color="#06d6a0", linewidth=2)
plt.plot(t, v, label="Actual Speed [m/s]", color="#118ab2", linewidth=2)
plt.title("Speed Tracking", fontsize=14, fontweight='bold')
plt.xlabel("Time [s]", fontsize=12)
plt.ylabel("Speed [m/s]", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# === --- PLOT 3: Actual vs Commanded Steering --- ===
plt.figure(figsize=(10, 5))
plt.plot(t, delta_actual, label="Actual Steering [deg]", color="#073b4c", linewidth=2)
plt.plot(t, delta_cmd, "--", label="Commanded Steering [deg]", color="#ffd166", linewidth=2)
plt.title("Steering Angle Tracking", fontsize=14, fontweight='bold')
plt.xlabel("Time [s]", fontsize=12)
plt.ylabel("Steering Angle [deg]", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.show()

# === Draw driven path ===
#planner.draw_path(np.column_stack((planner.x_ref, planner.y_ref)), color=(255, 0, 0), thickness=2)
#path_xy = np.column_stack((x, y))
#planner.draw_path(np.array([path_xy], dtype=np.float32), color=(0, 255, 255), thickness=3)
#planner.show_map_resized(roi_height_ratio=0.55, roi_width_ratio=0.35, scale=0.5)
#cv.waitKey(0)
#cv.destroyAllWindows()





cv.destroyAllWindows()
