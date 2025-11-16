#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import helper_functions as hf
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

data["t"] -= data["t"].iloc[0]
# === Extract signals ===
x = data["x"].to_numpy()
y = data["y"].to_numpy()
yaw = data["yaw"].to_numpy()
t = data["t"].to_numpy()
v = data["v"].to_numpy()
v_cmd = data["v_cmd"].to_numpy()
idx = data["idx"].to_numpy()

yaw = np.unwrap(yaw)
delta_actual = np.deg2rad(data["delta_actual"].to_numpy())
delta_cmd = data["delta_cmd"].to_numpy()
a_cmd = data["a_cmd"].to_numpy()
e_y = data["e_y"].to_numpy()
e_psi = data["e_psi"].to_numpy()




nodes_to_visit = [73, 97, 125, 150,135] # random area
#nodes_to_visit = [397, 200]      # round about 3rd exit - going highway
#nodes_to_visit = [397, 307, 377] # round about 1 then 2nd exit

traj_gen = TrajectoryGeneration(ds=0.01, N_horizon=30, v_max=0.6, v_min=0.2, use_curvature_velocity=True, smooth_velocity=True)
# Draw points on path
traj_gen.planner.draw_path_nodes(traj_gen.planner.route_list)
# Draw whole path
full_path = np.column_stack((traj_gen.planner.x_ref, traj_gen.planner.y_ref))
traj_gen.planner.draw_path(full_path)

#path by car
path_xy = np.column_stack((x, y))
traj_gen.planner.draw_path(np.array([path_xy], dtype=np.float32), color=(0, 255, 255), thickness=3)

# --- Draw start (green) and end (red) points ---
start = path_xy[0]
end = path_xy[-1]

traj_gen.planner.draw_points([start], color=(0, 255, 0), size=10)  # green = start
traj_gen.planner.draw_points([end], color=(0, 0, 255), size=10)    # red = end
cv.putText(traj_gen.planner.map, "Start point", hf.mR2pix(start), cv.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 3, cv.LINE_AA)
cv.putText(traj_gen.planner.map, "End point", hf.mR2pix(end), cv.FONT_HERSHEY_SIMPLEX, 1.8, (0,0,255), 3, cv.LINE_AA)

# Show map
traj_gen.planner.show_map_resized(roi_height_ratio=0.8, roi_width_ratio=0.4, scale=0.3)
cv.waitKey(0)
idx = np.array(idx, dtype=int)
s_ref, x_ref, y_ref, psi_ref, kappa_ref, _= traj_gen.planner.generate_path_passing_through(nodes_to_visit)
s_ref= s_ref[idx]
x_ref = x_ref[idx]
y_ref= y_ref[idx]
psi_ref = psi_ref[idx]
psi_ref = np.unwrap(psi_ref)
kappa_ref= kappa_ref[idx]
v_ref =0.4* np.ones(t.shape)

# === --- PLOT 1: Errors (lateral + heading) --- ===
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# --- Subplot 1: Lateral Error ---
axs[0].plot(t, e_y, color="C0", linewidth=2) #, label=r'Lateral Error $e_y \; [\mathrm{m}]$'
axs[0].set_ylabel(r'$e_y \; [\mathrm{m}]$', fontsize=12)
axs[0].set_title("Lateral Tracking Error", fontsize=14, fontweight='bold')
axs[0].legend(fontsize=11, loc='lower right')
axs[0].grid(True, linestyle="--", alpha=0.6)

# --- Subplot 2: Heading Error ---
axs[1].plot(t, e_psi,  color="C1", linewidth=2) #label=r'Heading Error $e_{\psi} \; [\mathrm{rad}]$',
axs[1].set_xlabel(r'Time [$\mathrm{s}$]', fontsize=12)
axs[1].set_ylabel(r'$e_{\psi} \; [\mathrm{rad}]$', fontsize=12)
axs[1].set_title("Heading Tracking Error", fontsize=14, fontweight='bold')
#axs[1].legend(fontsize=11, loc='lower right')
axs[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()


# === --- PLOT 2: Control Input Commands --- ===
fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# --- Subplot 1: Acceleration Command ---
axs[0].plot(t, a_cmd, label=r'$a_{\mathrm{cmd}}$', color="C2", linewidth=2)
axs[0].set_ylabel(r'$a_{\mathrm{cmd}} \; [\frac{m}{s^2}]$', fontsize=12)
axs[0].set_title("Acceleration Command", fontsize=14, fontweight='bold')
axs[0].legend(fontsize=11)
axs[0].grid(True, linestyle="--", alpha=0.6)

# --- Subplot 2: Steering Command ---
axs[1].plot(t, delta_cmd, label=r'$\delta_{\mathrm{cmd}}$', color="C3", linewidth=2)
axs[1].plot(t, delta_actual, "--", label=r'$\delta_{\mathrm{applied}}$', color="#ffd166", linewidth=1.5)
axs[1].set_ylabel(r'$\delta_{\mathrm{cmd}} \; [rad]$', fontsize=12)
axs[1].set_title("Steering Command", fontsize=14, fontweight='bold')
axs[1].legend(fontsize=11)
axs[1].grid(True, linestyle="--", alpha=0.6)

# --- Subplot 3: Speed Command ---
axs[2].plot(t, v_cmd, "--", label=r'$v_{\mathrm{cmd}}$', color="#06d6a0", linewidth=2)
axs[2].plot(t, v, label=r'$v_{\mathrm{applied}}$', color="#118ab2", linewidth=2)
#axs[2].plot(t, v_ref, '--', label=r'$v_{\mathrm{ref}}$', color="#ffd166", linewidth=1.5)
axs[2].set_xlabel(r'Time [$\mathrm{s}$]', fontsize=12)
axs[2].set_ylabel(r'$v_{\mathrm{cmd}} \; [\frac{m}{s}]$', fontsize=12)
axs[2].set_title("Speed Tracking", fontsize=14, fontweight='bold')
axs[2].legend(fontsize=11, loc='lower right')
axs[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()


# === --- PLOT 3: States --- ===
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# --- Subplot 1: X and Y Coordinates ---
axs[0].plot(t, x, label=r'$x$', color="#3498DB", linewidth=2)
axs[0].plot(t, x_ref, '--', label=r'$x_{\mathrm{ref}} \, y_{\mathrm{ref}}$', color="#ffd166", linewidth=1.5)
axs[0].plot(t, y, label=r'$y$', color="#808000", linewidth=2)
axs[0].plot(t, y_ref, '--', color="#ffd166", linewidth=1.5)
axs[0].set_ylabel(r'$X,\, Y \; [\mathrm{m}]$', fontsize=12)
axs[0].set_title("Position Coordinates", fontsize=14, fontweight='bold')
axs[0].legend(fontsize=11, loc='upper left')
axs[0].grid(True, linestyle="--", alpha=0.6)

# --- Subplot 2: Yaw Angle ---
axs[1].plot(t, yaw, label=r'$\theta$', color="#A52A2A", linewidth=2)
axs[1].plot(t, psi_ref, '--', label=r'$\theta_{\mathrm{ref}}$', color="#ffd166", linewidth=1.5)
axs[1].set_xlabel(r'Time [$\mathrm{s}$]', fontsize=12)
axs[1].set_ylabel(r'$\theta \; [\mathrm{rad}]$', fontsize=12)
axs[1].set_title("Direction Angle", fontsize=14, fontweight='bold')
axs[1].legend(fontsize=11, loc='upper left')
axs[1].grid(True, linestyle="--", alpha=0.6)

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