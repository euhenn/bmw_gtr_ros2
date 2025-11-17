#!/usr/bin/python3
import os
import cv2 as cv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pyclothoids import Clothoid
import helper_functions as hf


class PathPlanning:
    def __init__(self, map_img=cv.imread('data/2024_VerySmall.png')):
        """Initialize the path planner with a graph, map image, and configuration."""
        self.map = map_img

        # Default nodes (BFMC_2024 example)
        self.source = str(472)
        self.target = str(468)

        self.navigator = []      # Instructions for the route
        self.step_length = 0.01  # [m] step length for interpolation
        
        self.s_ref     = []
        self.x_ref     = []
        self.y_ref     = []
        self.psi_ref   = []
        self.kappa_ref = []
        self.clothoids = []

        # Load graph
        base_dir = os.path.dirname(__file__)
        graph_path = os.path.join(base_dir, 'data', 'final_graph.graphml')
        self.G = nx.read_graphml(graph_path)

        # Route-related data
        self.route_graph = nx.DiGraph()
        self.route_list = []    # list of nodes trough whics our path passes [(x, y, yaw), ...]

        # Extract node/edge info
        self.nodes_data = self.G.nodes.data()
        self.edges_data = self.G.edges.data()

        # Load intersection sets
        self.intersection_cen = list(np.loadtxt(os.path.join(base_dir, 'data', 'int_mid.txt'), dtype=str))
        self.roundabout = list(np.loadtxt(os.path.join(base_dir, 'data', 'round_in.txt'), dtype=str))
        self.intersection_in = list(np.loadtxt(os.path.join(base_dir, 'data', 'int_in.txt'), dtype=str))
        self.intersection_out = list(np.loadtxt(os.path.join(base_dir, 'data', 'int_out.txt'), dtype=str))
        self.stoplines = list(np.loadtxt(os.path.join(base_dir, 'data', 'stop_lines.txt'), dtype=str))


    # -------------------------------------------------------------
    # ROUTE GRAPH LOGIC
    # -------------------------------------------------------------
    def intersection_navigation(self, prev_node, curr_node, next_node):
        """Handle transitions through intersections."""
        prev_node = curr_node
        curr_node = next_node

        if curr_node != self.target:
            next_node = list(self.route_graph.successors(curr_node))[0]
        else:
            return prev_node, curr_node, None

        prev_node = curr_node
        curr_node = next_node
        next_node = list(self.route_graph.successors(curr_node))[0] if curr_node != self.target else None

        self.navigator.append(f"exit intersection at {curr_node}")
        self.navigator.append("go straight")
        return prev_node, curr_node, next_node
    
    def compute_route_list(self, mode="original", offset_dist=0.3):
        """
        Generate the route list (x, y, yaw) based on the route graph.

        Args:
            mode (string):
            - "original":    intersection-start points and their offset points.
            - "skip_center": offset points but remove intersection-start points. 
                For 'next_is_inter' use previous segment heading for the offset.
                For 'prev_is_inter' use outgoing segment heading for the offset.

            offset_dist (float): distance [m] to offset and create points in intersection for better curve pathing 
        """
        curr_node = self.source
        prev_node = curr_node
        next_node = list(self.route_graph.successors(curr_node))[0]
        self.route_list = []
        self.navigator.append("go straight")

        last_heading = None  # heading of the last appended point

        while curr_node != self.target:
            xp, yp = self.get_coord(prev_node)
            xc, yc = self.get_coord(curr_node)
            xn, yn = self.get_coord(next_node)

            # Incoming and outgoing headings at the CURRENT node
            heading_in  = np.rad2deg(np.arctan2(yc - yp, xc - xp))     # prev -> curr
            heading_out = np.rad2deg(np.arctan2(yn - yc, xn - xc))     # curr -> next

            next_is_inter = next_node in self.intersection_cen
            prev_is_inter = prev_node in self.intersection_cen
            curr_is_roundabout = curr_node in self.roundabout

            if next_is_inter:
                # approaching an intersection: place an EXIT offset after current node towards the intersection
                dx_in, dy_in = xc - xp, yc - yp

                if mode == "original":
                    # add center + offset (keep original behavior)
                    self.route_list.append((xc, yc, heading_in))
                    #self.route_list.append((xc + offset_dist * dx_in, yc + offset_dist * dy_in, heading_in))
                    last_heading = heading_in
                else:
                    # skip center — keep only the offset
                    smooth_heading = heading_in if last_heading is None else last_heading
                    self.route_list.append((xc + offset_dist * dx_in, yc + offset_dist * dy_in, smooth_heading))
                    last_heading = smooth_heading

                self.navigator.append(f"enter intersection near {curr_node}")
                # Navigate through the intersection to select the next edge
                prev_node, curr_node, next_node = self.intersection_navigation(prev_node, curr_node, next_node)
                continue

            elif prev_is_inter:
                # leaving an intersection: place an ENTRY offset before current node (toward the intersection)
                dx_out, dy_out = xn - xc, yn - yc  # direction of the NEXT segment (outgoing)

                if mode == "original":
                    # add entry offset (toward center) + the center itself
                    #self.route_list.append((xc - offset_dist * dx_out, yc - offset_dist * dy_out, heading_out))
                    self.route_list.append((xc, yc, heading_out))
                    last_heading = heading_out
                else:
                    # skip center — keep only the entry offset
                    # IMPORTANT FIX: use OUTGOING heading so it connects well to the next segment
                    self.route_list.append((xc - offset_dist * dx_out, yc - offset_dist * dy_out, heading_out))
                    last_heading = heading_out
            elif curr_is_roundabout:
                # Don t append those
                pass
            else:
                # normal segment (no intersection adjacent)
                dx, dy = xn - xp, yn - yp
                heading_mid = np.rad2deg(np.arctan2(dy, dx))
                self.route_list.append((xc, yc, heading_mid))
                last_heading = heading_mid

            # advance along the route
            prev_node, curr_node = curr_node, next_node
            if curr_node != self.target:
                next_node = list(self.route_graph.successors(curr_node))[0]
            else:
                # final target point: use outgoing heading from last step
                yaw_target = np.rad2deg(np.arctan2(yn - yc, xn - xc))
                self.route_list.append((xn, yn, yaw_target))
                next_node = None

        self.navigator.append("stop")



    def compute_shortest_path(self, source=None, target=None, step_length=0.01):
        """Compute shortest path between source and target nodes."""
        self.source = str(source) if source else self.source
        self.target = str(target) if target else self.target
        self.step_length = step_length

        route_nx = list(nx.shortest_path(self.G, source=self.source, target=self.target))
        self.route_graph = nx.DiGraph()
        self.route_graph.add_nodes_from(route_nx)

        for i in range(len(route_nx) - 1):
            self.route_graph.add_edge(route_nx[i], route_nx[i + 1], **self.G.get_edge_data(route_nx[i], route_nx[i + 1]))

        self.compute_route_list()

    # -------------------------------------------------------------
    # ROUTE INTERPOLATION
    # -------------------------------------------------------------
    @staticmethod
    def interpolate_route(route, step_length):
        """
        Interpolate a given route using Clothoid segments.

        Returns:
            s, x, y, psi, kappa, clothoid_list
        """
        s_all, x_all, y_all, psi_all, kappa_all = [], [], [], [], []
        clothoid_list = []
        total_s = 0.0

        for i in range(len(route) - 1):
            xc, yc, thc = route[i]
            xn, yn, thn = route[i + 1]
            thc, thn = np.deg2rad(thc), np.deg2rad(thn)

            clothoid = Clothoid.G1Hermite(xc, yc, thc, xn, yn, thn)
            clothoid_list.append(clothoid)

            length = clothoid.length
            n_pts = int(np.ceil(length / step_length)) + 1
            s_local = np.linspace(0.0, length, n_pts)

            x_local = [clothoid.X(s) for s in s_local]
            y_local = [clothoid.Y(s) for s in s_local]
            psi_local = [clothoid.Theta(s) for s in s_local]
            kappa_local = [clothoid.KappaStart + clothoid.dk * s for s in s_local]

            s_all.extend(total_s + s_local)
            x_all.extend(x_local)
            y_all.extend(y_local)
            psi_all.extend(psi_local)
            kappa_all.extend(kappa_local)

            total_s += length

        # Remove near-duplicates, keeping all aligned
        s_clean, x_clean, y_clean, psi_clean, kappa_clean = [], [], [], [], []
        prev_x, prev_y = None, None
        for s, x, y, psi, k in zip(s_all, x_all, y_all, psi_all, kappa_all):
            if prev_x is None or not (np.isclose(x, prev_x, rtol=1e-5) and np.isclose(y, prev_y, rtol=1e-5)):
                s_clean.append(s)
                x_clean.append(x)
                y_clean.append(y)
                psi_clean.append(psi)
                kappa_clean.append(k)
                prev_x, prev_y = x, y

        return (np.array(s_clean, dtype=np.float32),
                np.array(x_clean, dtype=np.float32),
                np.array(y_clean, dtype=np.float32),
                np.array(psi_clean, dtype=np.float32),
                np.array(kappa_clean, dtype=np.float32),
                clothoid_list)

    # -------------------------------------------------------------
    # PATH GENERATION
    # -------------------------------------------------------------
    def generate_path_passing_through(self, list_of_nodes, step_length=0.01):
        """Generate and store a continuous path through multiple nodes."""
        assert len(list_of_nodes) >= 2, "Need at least two nodes."

        full_route = []
        for i in range(len(list_of_nodes) - 1):
            src, tgt = list_of_nodes[i], list_of_nodes[i + 1]
            self.compute_shortest_path(source=src, target=tgt, step_length=step_length)
            segment_route = self.route_list.copy()
            full_route.extend(segment_route if i == 0 else segment_route[1:])

        # Remove duplicates & interpolate
        self.route_list = self._remove_route_duplicates(full_route)

        # Store references internally
        (
            self.s_ref,
            self.x_ref,
            self.y_ref,
            self.psi_ref,
            self.kappa_ref,
            self.clothoids,
        ) = PathPlanning.interpolate_route(self.route_list, step_length)

        #print(f"[INFO] Path generated: {len(self.s_ref)} points, {len(self.clothoids)} clothoids")
        return self.s_ref, self.x_ref, self.y_ref, self.psi_ref, self.kappa_ref, self.clothoids


    @staticmethod
    def _remove_route_duplicates(route):
        """Remove duplicate route entries."""
        clean_route, prev = [], None
        for x, y, yaw in route:
            if prev is None or np.linalg.norm(np.array([x, y]) - np.array(prev)) > 1e-3:
                clean_route.append((x, y, yaw))
                prev = (x, y)
        return clean_route

    # -------------------------------------------------------------
    # VISUALIZATION & METHODS
    # -------------------------------------------------------------
    def get_coord(self, node):
        """Return metric coordinates of a graph node."""
        return hf.mL2mR(np.array([self.nodes_data[node]['x'], self.nodes_data[node]['y']]))
        
    @staticmethod
    def project_point_on_path(X, Y, clothoids):
        """
        Find the closest projection of a point (X, Y) onto a sequence of clothoids.

        Args:
            X, Y (float): point coordinates in meters
            clothoids (list): list of Clothoid objects from interpolate_route()

        Returns:
            (x_proj, y_proj): coordinates of the closest point on the path
            s_global (float): arc length along the full path where the projection lies
            dist_min (float): minimum distance between (X, Y) and the path
            idx_clothoid (int): index of the closest clothoid segment
        """
        best_x, best_y, s_global, dist_min = None, None, 0.0, float("inf")
        s_accum = 0.0
        idx_clothoid = -1

        for i, clothoid in enumerate(clothoids):
            try:
                (x_proj, y_proj), s_local, dist = clothoid.ProjectPointOntoClothoid(X, Y)
            except Exception as e:
                print(f"[Warning] Clothoid {i} projection failed: {e}")
                continue

            if dist < dist_min:
                dist_min = dist
                best_x, best_y = x_proj, y_proj
                s_global = s_accum + s_local
                idx_clothoid = i

            s_accum += clothoid.length

        return (best_x, best_y), s_global, dist_min, idx_clothoid
    
    def time2space(self, X, Y, YAW):
        """
        Convert (x, y, yaw) to spatial coordinates (s, epsi, ey) along the stored reference path.
        Requires self.s_ref, self.psi_ref, and self.clothoids to be available.
        """
        try:
            # Ensure reference data exists
            if not hasattr(self, "clothoids") or self.clothoids is None:
                raise ValueError("Reference path not initialized. Call generate_path_passing_through() first.")

            # Project the current position onto the clothoid path
            (x_proj, y_proj), s_proj, _ , _ = self.project_point_on_path(X, Y, self.clothoids)

            # Find index of closest point
            idx = int(np.argmin(np.abs(self.s_ref - s_proj)))

            # Reference heading at projection
            psi0 = self.psi_ref[idx]

            # Lateral deviation (cross-track error)
            #ey = np.cos(psi0) * (Y - y_proj) - np.sin(psi0) * (X - x_proj)
            # Left side is positive, right is negative
            ey = - (np.cos(psi0) * (Y - y_proj) - np.sin(psi0) * (X - x_proj))

            # Heading error, wrapped to [-π, π]
            epsi = psi0 - YAW
            epsi = (epsi + np.pi) % (2 * np.pi) - np.pi

            return s_proj, epsi, ey, idx

        except Exception as e:
            print(f"[Error] time2space failed: {e}")
            return None, None, None, None

    def show_map_resized(self, roi_height_ratio=1, roi_width_ratio=1, scale=0.2, window_name="Planned Path"):
        """
        Display a resized region of interest (ROI) of the map.

        Args:
            roi_height_ratio (float): fraction of map height to keep (from bottom).
            roi_width_ratio (float): fraction of map width to keep (from left).
            scale (float): resize scaling factor.
            window_name (str): name of the OpenCV window.

        Returns:
            np.ndarray: the resized ROI image.
        """
        if self.map is None:
            raise ValueError("Map image not loaded.")

        h, w = self.map.shape[:2]
        roi = self.map[int(h * (1 - roi_height_ratio)):, :int(w * roi_width_ratio)]
        resized = cv.resize(roi, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

        cv.imshow(window_name, resized)
        return resized

        
    def draw_path_nodes(self, route):
        """Draw path nodes, from the route list, on the map."""
        for x, y, _ in route:
            cv.circle(self.map, hf.mR2pix(np.array([x, y])), 10, (255, 0, 0), -1)

    def draw_path(self, path_xy, color=(200, 200, 0), thickness=4):
        """Draw route path"""
        cv.polylines(self.map, [hf.mR2pix(path_xy)], isClosed=False, color=color, thickness=thickness, lineType=cv.LINE_AA)

    def draw_path_gradient(self, path_xy, values, thickness=5):
        """
        Draw a path with a color gradient (red to green) based on values (e.g., velocity).

        Args:
            path_xy (array-like): Nx2 array of (x, y) metric coordinates [m].
            values (array-like): N array of scalar values (e.g., velocity).
            thickness (int): line thickness in pixels.
        """
        if self.map is None:
            raise ValueError("Map image not loaded.")
        if len(path_xy) != len(values):
            raise ValueError("path_xy and values must have the same length.")

        # Convert metric to pixel coordinates
        pts_pix = np.array([hf.mR2pix(np.array(p)) for p in path_xy], dtype=np.int32)

        # Set min/max normalization range
        vmin = np.min(values) 
        vmax = np.max(values) 
        vmax = vmax if vmax != vmin else vmax + 1e-6  # avoid div by zero

        # Normalize to [0, 1]
        norm_vals = (values - vmin) / (vmax - vmin)

        # Draw segment by segment
        for i in range(len(pts_pix) - 1):
            t = norm_vals[i]
            # Red (low) to Green (high)
            color = (
                0,                   # B
                int(255 * t),        # G
                int(255 * (1 - t))   # R 
            )
            cv.line(self.map, tuple(pts_pix[i]), tuple(pts_pix[i + 1]), color, thickness, cv.LINE_AA)


    def draw_points(self, points, shape="circle", size=15, color=(0, 255, 0), thickness=-1):
        """
        Draw a list of points on the map image.

        Args:
            points (list): list of (x, y) points in metric coordinates [m].
            shape (str): "circle" or "square".
            size (int): radius (circle) or half-side (square) in pixels.
            color (tuple): BGR color for the points, e.g., (0, 255, 0).
            thickness (int): OpenCV thickness (-1 for filled shapes).
            show (bool): if True, show the map after drawing.
        """
        if self.map is None:
            raise ValueError("Map image is not loaded.")

        for x, y in points:
            pix = hf.mR2pix(np.array([x, y]))
            if shape.lower() == "circle":
                cv.circle(self.map, pix, size, color, thickness)
            elif shape.lower() == "square":
                cv.rectangle(self.map,
                             (pix[0] - size, pix[1] - size),
                             (pix[0] + size, pix[1] + size),
                             color, thickness)
            else:
                raise ValueError(f"Unsupported shape '{shape}'. Use 'circle' or 'square'.")
            
    def draw_car(self, x, y, yaw, length=0.4, width=0.2, color_body=(0, 0, 255), color_dir=(0, 255, 255), thickness=2):
        """
        Draw a rectangular car with a direction line on the map.

        Args:
            x, y (float): car position in metric coordinates [m].
            yaw (float): car heading [rad].
            length (float): car length [m].
            width (float): car width [m].
            color_body (tuple): BGR color for the car body.
            color_dir (tuple): BGR color for the direction line.
            thickness (int): OpenCV line thickness.
        """
        if self.map is None:
            raise ValueError("Map image not loaded.")

        # --- Compute rectangle corners in local coordinates (centered at car) ---
        half_L = length / 2.0
        half_W = width / 2.0

        # Local corners (front-left, front-right, rear-right, rear-left)
        corners_local = np.array([
            [ half_L,  half_W],
            [ half_L, -half_W],
            [-half_L, -half_W],
            [-half_L,  half_W]
        ])

        # Rotation matrix from yaw
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw),  np.cos(yaw)]])

        # Transform corners to world frame
        corners_world = (R @ corners_local.T).T + np.array([x, y])

        # Convert to pixel coordinates
        corners_pix = np.array([hf.mR2pix(p) for p in corners_world], dtype=np.int32)

        # Draw the car rectangle
        cv.polylines(self.map, [corners_pix.reshape((-1, 1, 2))], isClosed=True, color=color_body, thickness=thickness)

        # --- Draw the heading line (front direction) ---
        front_center = (R @ np.array([half_L, 0.0])) + np.array([x, y])
        front_pix = hf.mR2pix(front_center)
        center_pix = hf.mR2pix(np.array([x, y]))

        cv.line(self.map, center_pix, front_pix, color=color_dir, thickness=thickness)

    @staticmethod
    def plot_path_and_curvature(x, y, kappa, s=None, route=None, map_img=cv.imread('data/2024_VerySmall.png'), zoom_margin=100):
        """
        Plot the path (XY) and curvature vs arc-length.

        Args:
            x, y (array): path coordinates [m]
            kappa (array): curvature values [1/m]
            s (array): arc-lengths [m]
            route (list): list of (x, y, yaw) waypoints
            map_img (np.ndarray): background map image
            zoom_margin (int): pixel margin to include around route area for zoom
        """
        path = np.column_stack((x, y))
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

        # === TOP: Path on map ===
        ax_path = fig.add_subplot(gs[0, 0])
        img_rgb = cv.cvtColor(map_img.copy(), cv.COLOR_BGR2RGB)
        ax_path.imshow(img_rgb, origin='upper')

        # Path and waypoints in pixel coordinates
        pix_pts = np.array([hf.mR2pix(p) for p in path])
        ax_path.plot(pix_pts[:, 0], pix_pts[:, 1], linewidth=2, color='yellow', label='Path (pixels)')

        if route:
            route_pix = np.array([hf.mR2pix(np.array([x, y])) for (x, y, _) in route])
            ax_path.scatter(route_pix[:, 0], route_pix[:, 1], c='red', s=25, label='Waypoints')

        # === Automatic zoom around route ===
        x_min, y_min = np.min(pix_pts, axis=0)
        x_max, y_max = np.max(pix_pts, axis=0)

        # Add a margin to make the view more comfortable
        ax_path.set_xlim(x_min - zoom_margin, x_max + zoom_margin)
        ax_path.set_ylim(y_max + zoom_margin, y_min - zoom_margin)  # flipped Y because imshow origin='upper'

        ax_path.set_title("Path on Map (zoomed on route)")
        ax_path.axis('off')
        ax_path.legend()

        # === BOTTOM: Curvature plot ===
        ax_k = fig.add_subplot(gs[1, 0])
        if s is None:
            diffs = np.diff(path, axis=0)
            seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
            s = np.zeros(len(path), dtype=np.float32)
            s[1:] = np.cumsum(seg_lengths)
        ax_k.plot(s, kappa, linewidth=1.5, label='Curvature κ(s)')
        ax_k.set_xlabel('Arc length s [m]')
        ax_k.set_ylabel('Curvature [1/m]')
        ax_k.grid(True)
        ax_k.legend()

        plt.show()



# -------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------

if __name__ == "__main__":
    map_img = cv.imread('data/2024_VerySmall.png')
    planner = PathPlanning(map_img)
    nodes_to_pass = [73, 97, 125, 150, 135]
    #nodes_to_pass = [330, 337]
    #nodes_to_pass = [397, 307, 377]

    # Generate path (now auto-stored inside planner)
    planner.generate_path_passing_through(nodes_to_pass, step_length=0.01)
    s_ref, x_ref, y_ref, psi_ref, kappa_ref = planner.s_ref, planner.x_ref, planner.y_ref, planner.psi_ref, planner.kappa_ref


    # Example car state
    x, y, yaw = 2.5, 4.2, np.deg2rad(280)
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi  # wrap yaw

    # Convert to spatial domain (no more s_ref, psi_ref, clothoids args)
    s, epsi, ey, idx = planner.time2space(x, y, yaw)
    print(f"s = {s:.3f} m, ey = {ey:.3f} m, epsi = {epsi:.2f} rad = {np.rad2deg(epsi):.2f} deg, idx = {idx}")

    # Draw and visualize
    planner.draw_path_nodes(planner.route_list)
    planner.draw_path(np.column_stack((planner.x_ref, planner.y_ref)))
    planner.draw_car(x, y, yaw)
    planner.draw_car(x_ref[idx], y_ref[idx], psi_ref[idx])

    planner.show_map_resized(roi_height_ratio=0.55,roi_width_ratio=0.35,scale=0.5)
    cv.waitKey(0)

    planner.plot_path_and_curvature(x_ref, y_ref, kappa_ref, s_ref, route=planner.route_list, map_img=map_img)
    
    cv.destroyAllWindows()