import numpy as np
import cvxpy as cp
import scipy.interpolate as si
import matplotlib.pyplot as plt


# =====================================================
# 1.  Spline utilities
# =====================================================
def build_spline(control_pts):
    pts = np.asarray(control_pts)
    chords = np.r_[0, np.cumsum(np.sqrt(((pts[1:] - pts[:-1])**2).sum(axis=1)))]
    chords /= chords[-1] if chords[-1] > 0 else 1
    tck_x = si.CubicSpline(chords, pts[:, 0], bc_type="natural")
    tck_y = si.CubicSpline(chords, pts[:, 1], bc_type="natural")
    return tck_x, tck_y


def discretize_spline(tck_x, tck_y, n=400):
    t = np.linspace(0, 1, 2000)
    x, y = tck_x(t), tck_y(t)
    dx, dy = tck_x(t, 1), tck_y(t, 1)
    ddx, ddy = tck_x(t, 2), tck_y(t, 2)
    ds_dt = np.sqrt(dx**2 + dy**2)
    s = np.cumsum(np.r_[0, 0.5*(ds_dt[1:]+ds_dt[:-1])*np.diff(t)])
    s /= s[-1]
    s_uniform = np.linspace(0, 1, n)
    t_s = np.interp(s_uniform, s, t)
    x, y = tck_x(t_s), tck_y(t_s)
    dx, dy = tck_x(t_s, 1), tck_y(t_s, 1)
    ddx, ddy = tck_x(t_s, 2), tck_y(t_s, 2)
    kappa = (dx*ddy - dy*ddx) / np.maximum((dx**2 + dy**2)**1.5, 1e-9)
    return dict(x=x, y=y, kappa=kappa, s=np.linspace(0, 1, n))


# =====================================================
# 2.  Convex speed-profile solver
# =====================================================
def solve_speed_profile(geom,
                        a_lat_max=8.0,
                        a_t_max=2.0,
                        a_t_min=-6.0,
                        v_initial=0.1,
                        v_final=0.1):

    s = geom["s"]
    N = len(s)
    ds = np.diff(s)
    kappa = np.abs(geom["kappa"])
    v_max_sq = np.where(kappa < 1e-8, 1e6, a_lat_max / np.maximum(kappa, 1e-9))

    w = cp.Variable(N, nonneg=True)
    ds_node = np.r_[ds, ds[-1]]

    obj = cp.Minimize(cp.sum(cp.multiply(ds_node, cp.power(w, -0.5))))
    cons = [w <= v_max_sq, w >= 1e-4]

    for i in range(N-1):
        cons += [w[i+1] - w[i] <= 2*a_t_max*ds[i],
                 w[i+1] - w[i] >= 2*a_t_min*ds[i]]

    cons += [w[0] == v_initial**2, w[-1] == v_final**2]
    prob = cp.Problem(obj, cons)

    # Prefer Clarabel (new default), fallback to SCS
    try:
        prob.solve(solver=cp.CLARABEL, verbose=True)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=True)

    if w.value is None:
        raise RuntimeError("Solver failed")

    wv = np.maximum(w.value, 1e-6)
    v = np.sqrt(wv)
    total_time = float(np.sum(ds_node / np.maximum(v, 1e-6)))
    return v, total_time


# =====================================================
# 3.  Box-based control point sampler
# =====================================================
def sample_control_vectors(boxes, n_samples=100):
    """
    Sample random control points inside each oriented square box.
    Each box is a dict with fields: center, angle, corners.
    """
    all_pts = []
    for _ in range(n_samples):
        pts = []
        for b in boxes:
            cx, cy = b["center"]
            theta = b["angle"]
            half_size = np.linalg.norm(b["corners"][0] - b["corners"][1]) / 2.0

            # random local coordinates within [-half_size, half_size]
            lx = np.random.uniform(-half_size, half_size)
            ly = np.random.uniform(-half_size, half_size)

            # rotate by angle and translate to global position
            x = cx + lx * np.cos(theta) - ly * np.sin(theta)
            y = cy + lx * np.sin(theta) + ly * np.cos(theta)
            pts.append([x, y])

        all_pts.append(np.array(pts))
    return all_pts



def search_best_path(boxes, n_samples=100):
    best_T, best_data = np.inf, None
    for ctrl in sample_control_vectors(boxes, n_samples):
        try:
            tck_x, tck_y = build_spline(ctrl)
            geom = discretize_spline(tck_x, tck_y)
            v, T = solve_speed_profile(geom)
            if T < best_T and np.isfinite(T):
                best_T = T
                best_data = (ctrl, geom, v, T)
        except Exception:
            continue
    return best_data


# =====================================================
# 4.  Automatic box generation along path
# =====================================================
def generate_oriented_boxes_along_path(path, num_boxes=5, box_size=2.0, spacing=None):
    """
    Generate oriented square boxes along a path with specified spacing and rotation.
    
    Returns:
    - boxes: list of dicts {center: (x, y), angle: theta, corners: (4x2 array)}
    """
    path = np.asarray(path)

    # Compute cumulative distances
    distances = np.zeros(len(path))
    for i in range(1, len(path)):
        distances[i] = distances[i-1] + np.linalg.norm(path[i] - path[i-1])
    total_length = distances[-1]

    # Determine where to place boxes
    if spacing is not None:
        num_boxes = max(2, int(total_length / spacing))
    target_distances = np.linspace(0, total_length, num_boxes)

    half_size = box_size / 2.0
    boxes = []

    for dist in target_distances:
        # Find segment containing this distance
        idx = np.searchsorted(distances, dist) - 1
        idx = np.clip(idx, 0, len(path) - 2)

        # Interpolate position
        t = (dist - distances[idx]) / (distances[idx + 1] - distances[idx])
        center = path[idx] + t * (path[idx + 1] - path[idx])

        # Compute local tangent direction
        tangent = path[idx + 1] - path[idx]
        angle = np.arctan2(tangent[1], tangent[0])  # radians

        # Compute rotated square corners
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        base = np.array([[-half_size, -half_size],
                         [ half_size, -half_size],
                         [ half_size,  half_size],
                         [-half_size,  half_size]])
        corners = (R @ base.T).T + center

        boxes.append({
            "center": center,
            "angle": angle,
            "corners": corners
        })

    return boxes


# =====================================================
# 5.  Example main
# =====================================================
if __name__ == "__main__":
    import raceline as rl
    import cv2 as cv
    import helper_functions as hf  # assuming this has mR2pix()

    # Load map
    map_img = cv.imread('data/2024_VerySmall.png')

    path_planning = rl.PathPlanning(map_img, show_imgs=False)
    nodes_to_pass = [140, 14]
    path_planning.generate_path_passing_through_full(nodes_to_pass, step_length=0.01, method='spline')

    path = path_planning.path  # np.array([[x, y], ...]) in meters

    # Generate oriented boxes along the path
    boxes = generate_oriented_boxes_along_path(path, spacing=0.30, box_size=0.25)

    # === Visualization ===
    fig, ax_path = plt.subplots(figsize=(10, 10))

    # ---- Show map as background ----
    img_rgb = cv.cvtColor(map_img.copy(), cv.COLOR_BGR2RGB)
    ax_path.imshow(img_rgb, origin='upper')

    # ---- Convert and plot path ----
    pix_pts = np.array([hf.mR2pix(p) for p in path])
    ax_path.plot(pix_pts[:, 0], pix_pts[:, 1], 'g-', lw=2, alpha=0.7, label='Path (pixels)')

    # ---- Plot oriented boxes ----
    for i, b in enumerate(boxes):
        # Convert box corners from meters → pixels
        pix_corners = np.array([hf.mR2pix(p) for p in b["corners"]])
        pix_corners = np.vstack([pix_corners, pix_corners[0]])  # close the loop
        ax_path.plot(pix_corners[:, 0], pix_corners[:, 1], 'r-', lw=1.2, alpha=0.8)

    # ---- Final formatting ----
    # compute bounding box of the path
    min_x, max_x = pix_pts[:, 0].min(), pix_pts[:, 0].max()
    min_y, max_y = pix_pts[:, 1].min(), pix_pts[:, 1].max()

    # add a little margin
    margin = 100  # pixels
    ax_path.set_xlim(min_x - margin, max_x + margin)
    ax_path.set_ylim(max_y + margin, min_y - margin)  # inverted y


    ax_path.set_title("Zoomed Bottom-Left Corner with Path and Oriented Boxes")
    ax_path.axis('off')
    ax_path.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Solve for best path
    # -------------------------
    result = search_best_path(boxes, n_samples=200)

    if result is None:
        print("No feasible spline found from sampled control points.")
    else:
        ctrl, geom, v_profile, total_time = result
        print(f"Optimal time to traverse path: {total_time:.2f} seconds")

        # Convert geometry (meters) -> pixels for plotting
        geom_pts_m = np.vstack([geom["x"], geom["y"]]).T  # Nx2 in meters
        geom_pix = np.array([hf.mR2pix(p) for p in geom_pts_m])

        # Convert control points to pixels
        ctrl_pix = np.array([hf.mR2pix(p) for p in ctrl])

        # Reuse same zoom bounds computed earlier (so map view stays focused)
        fig2, ax2 = plt.subplots(figsize=(10, 10))
        ax2.imshow(img_rgb, origin='upper')

        # Plot zoom/limits to match previously used view
        ax2.set_xlim(ax_path.get_xlim())
        ax2.set_ylim(ax_path.get_ylim())

        # Plot the speed-colored trajectory (use scatter so color per point is visible)
        sc = ax2.scatter(geom_pix[:, 0], geom_pix[:, 1], c=v_profile, s=8, cmap="viridis",
                         edgecolors='none', label='Optimized trajectory (pixels)')
        cbar = plt.colorbar(sc, ax=ax2, fraction=0.036, pad=0.04)
        cbar.set_label("speed (m/s)")

        # Plot control points and connecting polyline (in pixel coords)
        ax2.plot(ctrl_pix[:, 0], ctrl_pix[:, 1], "k--", lw=1.2, label="control pts (meters -> pix)")
        ax2.scatter(ctrl_pix[:, 0], ctrl_pix[:, 1], c='yellow', s=30, zorder=5)

        # Re-plot oriented boxes (optional, for context)
        for i, b in enumerate(boxes):
            pix_corners = np.array([hf.mR2pix(p) for p in b["corners"]])
            pix_corners = np.vstack([pix_corners, pix_corners[0]])
            ax2.plot(pix_corners[:, 0], pix_corners[:, 1], 'r-', lw=1.0, alpha=0.7)

        ax2.set_title(f"Best path (T = {total_time:.2f} s) — speed colored")
        ax2.axis('off')
        ax2.legend(loc='lower right')
        plt.tight_layout()
        plt.show()


