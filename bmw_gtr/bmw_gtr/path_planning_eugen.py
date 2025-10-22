#!/usr/bin/python3
import networkx as nx
import numpy as np
import cv2 as cv
from pyclothoids import Clothoid
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import helper_functions as hf
import os



class PathPlanning:
    def __init__(self, map_img=cv.imread('data/2024_VerySmall.png'), show_imgs=False):
        
        # map to plot trajectory and car
        self.map = map_img
        self.show_imgs = show_imgs

        # start and end nodes default
        self.source = str(472)   # BFMC_2024
        self.target = str(468)   # BFMC_2024

        # initialize path
        self.path = []           # np.array([[x0,y0], [x1,y1], ...])
        self.path_curvature = [] # np.array([k0, k1, k2, ...])
        self.navigator = []      # set of instruction for the navigator
        self.step_length = 0.01  # [m] step length for interpolation

        base_dir = os.path.dirname(__file__)
        graph_path = os.path.join(base_dir, 'data', 'final_graph.graphml')
        self.G = nx.read_graphml(graph_path)

        # initialize route subgraph and list for interpolation
        self.route_graph = nx.DiGraph()
        self.route_list = []    # [(x0,y0,yaw0), (x1,y1,yaw1), ...]

        self.nodes_data = self.G.nodes.data()
        self.edges_data = self.G.edges.data()

        # load intersection files
        intersection_cen_path = os.path.join(base_dir, 'data', 'int_mid.txt')
        intersection_in_path = os.path.join(base_dir, 'data', 'int_in.txt')
        intersection_out_path = os.path.join(base_dir, 'data', 'int_out.txt')
        self.intersection_cen = list(np.loadtxt(intersection_cen_path, dtype=str))
        self.intersection_in = list(np.loadtxt(intersection_in_path, dtype=str))
        self.intersection_out = list(np.loadtxt(intersection_out_path, dtype=str))
        # Forbidden nodes to skip as start/end points
        self.forbidden_nodes = self.intersection_cen + self.intersection_in + self.intersection_out

        # import nodes and edges
        self.list_of_nodes = list(self.G.nodes)
        self.list_of_edges = list(self.G.edges)

        # Only valid starting positions
        all_start_nodes = list(self.G.nodes)
        self.all_start_nodes = []
        for n in all_start_nodes:
            if n in self.forbidden_nodes:
                continue
            else:
                self.all_start_nodes.append(n)
        
        self.all_nodes_coords = np.array([self.get_coord(node) for node in self.all_start_nodes])

    def intersection_navigation(self, prev_node, curr_node, next_node):
        prev_node = curr_node
        curr_node = next_node
        if curr_node != self.target:
            next_node = list(self.route_graph.successors(curr_node))[0]
        else:
            return prev_node, curr_node, None

        prev_node = curr_node
        curr_node = next_node
        if curr_node != self.target:
            next_node = list(self.route_graph.successors(curr_node))[0]
        else:
            next_node = None

        self.navigator.append("exit intersection at " + curr_node)
        self.navigator.append("go straight")
        return prev_node, curr_node, next_node

    def compute_route_list(self):
        """Augments the route stored in self.route_graph"""
        curr_node = self.source
        prev_node = curr_node
        next_node = list(self.route_graph.successors(curr_node))[0]

        # reset route list
        self.route_list = []

        self.navigator.append("go straight")
        while curr_node != self.target:
            pp = self.get_coord(prev_node)
            xp, yp = pp[0], pp[1]
            pc = self.get_coord(curr_node)
            xc, yc = pc[0], pc[1]
            pn = self.get_coord(next_node)
            xn, yn = pn[0], pn[1]

            next_is_intersection = next_node in self.intersection_cen
            prev_is_intersection = prev_node in self.intersection_cen

            if next_is_intersection:
                dx, dy = xc - xp, yc - yp
                self.route_list.append((xc, yc, np.rad2deg(np.arctan2(dy, dx))))
                self.route_list.append((xc + 0.3 * dx, yc + 0.3 * dy, np.rad2deg(np.arctan2(dy, dx))))
                self.navigator.append("enter intersection at " + curr_node)
                prev_node, curr_node, next_node = self.intersection_navigation(prev_node, curr_node, next_node)
                continue
            elif prev_is_intersection:
                dx, dy = xn - xc, yn - yc
                self.route_list.append((xc - 0.3 * dx, yc - 0.3 * dy, np.rad2deg(np.arctan2(dy, dx))))
                self.route_list.append((xc, yc, np.rad2deg(np.arctan2(dy, dx))))
            else:
                dx, dy = xn - xp, yn - yp
                self.route_list.append((xc, yc, np.rad2deg(np.arctan2(dy, dx))))

            prev_node = curr_node
            curr_node = next_node
            if curr_node != self.target:
                next_node = list(self.route_graph.successors(curr_node))[0]
            else:
                dx, dy = xn - xp, yn - yp
                self.route_list.append((xn, yn, np.rad2deg(np.arctan2(dy, dx))))
                next_node = None

        self.navigator.append("stop")

    def compute_shortest_path(self, source=None, target=None, step_length=0.01):
        """Generates the shortest path between source and target nodes using Clothoid interpolation"""
        src = str(source) if source is not None else self.source
        tgt = str(target) if target is not None else self.target
        self.source, self.target = src, tgt
        self.step_length = step_length

        # generate shortest route
        route_nx = list(nx.shortest_path(self.G, source=src, target=tgt))
        self.route_graph = nx.DiGraph()
        self.route_graph.add_nodes_from(route_nx)

        for i in range(len(route_nx) - 1):
            self.route_graph.add_edges_from([(route_nx[i], route_nx[i+1], self.G.get_edge_data(route_nx[i], route_nx[i+1]))])

        self.compute_route_list()

    def remove_route_duplicates(self):
        """Removes duplicates from route list"""
        prev_x, prev_y = 0, 0
        for i, (x, y, yaw) in enumerate(self.route_list):
            if np.linalg.norm(np.array([x, y]) - np.array([prev_x, prev_y])) < 0.001:
                self.route_list.pop(i)
            prev_x, prev_y = x, y


    def generate_path_passing_through(self, list_of_nodes, step_length=0.01, method='clothoid'):
        """Generate a continuous path and route list through multiple nodes."""
        assert len(list_of_nodes) >= 2, "List of nodes must have at least 2 nodes"
        
        # Initialize empty full path and route list
        full_path = []
        full_route_list = []

        for i in range(len(list_of_nodes) - 1):
            src, tgt = list_of_nodes[i], list_of_nodes[i+1]
            print(f"Segment {i+1}: {src} -> {tgt}")

            # Compute segment path
            self.compute_shortest_path(source=src, target=tgt, step_length=step_length)

            # Save route list before it gets overwritten next iteration
            segment_route = self.route_list.copy()
    
            if i == 0:
                full_route_list.extend(segment_route)
            else:
                # skip duplicate start point to avoid double node connection
                full_route_list.extend(segment_route[1:])

        # Store merged results and interpolate full path
        self.route_list = full_route_list
        self.remove_route_duplicates()
        if method == 'clothoid':
            self.path, self.path_curvature = PathPlanning.interpolate_route(self.route_list, step_length)
        elif method == 'spline':
            self.path, self.path_curvature, self.spline_x, self.spline_y = PathPlanning.interpolate_route_spline(self.route_list, step_length)
 

    def get_coord(self, node):
        x = self.nodes_data[node]['x']
        y = self.nodes_data[node]['y']
        p = np.array([x, y])
        return hf.mL2mR(p)

    @staticmethod
    def interpolate_route_spline(route, step_length=0.01):
        """Interpolate a given route using cubic splines."""
        route = np.array(route)
        X, Y = route[:, 0], route[:, 1]

        # parameterize by cumulative arc length
        s = np.zeros(len(X))
        s[1:] = np.cumsum(np.hypot(np.diff(X), np.diff(Y)))

        # fit cubic splines for X(s) and Y(s)
        csx = CubicSpline(s, X)
        csy = CubicSpline(s, Y)

        # sample along arc length using step_length
        total_length = s[-1]
        s_fine = np.arange(0, total_length + step_length, step_length)
        x_fine = csx(s_fine)
        y_fine = csy(s_fine)

        # optional curvature computation
        dx = csx(s_fine, 1)
        dy = csy(s_fine, 1)
        ddx = csx(s_fine, 2)
        ddy = csy(s_fine, 2)
        curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5

        return np.column_stack((x_fine, y_fine)), curvature, csx, csy


    @staticmethod
    def interpolate_route(route, step_length):
        """Interpolate a given route using Clothoid segments."""
        path_X, path_Y, path_K = [], [], []  # lists for x, y, curvature

        for i in range(len(route) - 1):
            xc, yc, thc = route[i]
            xn, yn, thn = route[i + 1]
            thc, thn = np.deg2rad(thc), np.deg2rad(thn)

            # Create clothoid segment
            clothoid = Clothoid.G1Hermite(xc, yc, thc, xn, yn, thn)
            length = clothoid.length
            n_pts = max(2, int(length / step_length))  # at least 2 points
            s_vals = np.linspace(0, length, n_pts)

            # Sample points along clothoid
            X, Y = clothoid.SampleXY(n_pts)

            # Compute curvature values: kappa(s) = KappaStart + dk * s
            kappa_start = clothoid.KappaStart
            dk = clothoid.dk
            K = kappa_start + dk * s_vals

            # Store all results
            path_X.extend(X)
            path_Y.extend(Y)
            path_K.extend(K)

        # Remove near-duplicate points
        path, curvature, prev_x, prev_y = [], [], 0.0, 0.0
        for x, y, k in zip(path_X, path_Y, path_K):
            if not (np.isclose(x, prev_x, rtol=1e-5) and np.isclose(y, prev_y, rtol=1e-5)):
                path.append([x, y])
                curvature.append(k)
            prev_x, prev_y = x, y

        return np.array(path, dtype=np.float32), np.array(curvature, dtype=np.float32)



    def draw_path(self):

        # draw all points in given path
        for point in self.route_list:
            x, y, _ = point
            p = np.array([x, y])
            cv.circle(self.map, hf.mR2pix(p), 10, (255, 0, 0), -1)
            #print(point)

        # draw trajectory
        cv.polylines(self.map, [hf.mR2pix(self.path)], False, (200, 200, 0), thickness=4, lineType=cv.LINE_AA)

        if self.show_imgs:
            # --- Crop the region that I want to zoom into ---
            h, w = self.map.shape[:2]
            crop_width = int(w * 0.35)   # take % of width
            crop_height = int(h * 0.55)  # take % of height
            roi = self.map[h - crop_height:h, 0:crop_width]

            # --- Scale down the cropped region for easier viewing ---
            scale_percent = 50
            width = int(roi.shape[1] * scale_percent / 100)
            height = int(roi.shape[0] * scale_percent / 100)
            resized_roi = cv.resize(roi, (width, height), interpolation=cv.INTER_AREA)

            cv.imshow("Path passing through the specified nodes", resized_roi)
            cv.waitKey(0)

    def plot_path_and_curvature(self, show_map=False, figsize=(10, 8)):
        """
        Plot the XY path and curvature vs arc-length.
        If show_map=True the map image is shown as a background and the path is overlaid (uses hf.mR2pix).
        """
        if not hasattr(self, 'path') or self.path is None or len(self.path) == 0:
            raise ValueError("Path has not been generated yet. Call generate_path_passing_through_full(...) first.")

        # ensure numpy arrays
        path = np.asarray(self.path)
        k = np.asarray(self.path_curvature)

        # compute arc-length (s)
        diffs = np.diff(path, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        s = np.zeros(len(path), dtype=np.float32)
        s[1:] = np.cumsum(seg_lengths)

        # create figure with two rows: path and curvature
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

        # --- Path plot (XY) ---
        ax_path = fig.add_subplot(gs[0, 0])
        if show_map and (self.map is not None):
            # show map as background (convert BGR to RGB for matplotlib)
            img = cv.cvtColor(self.map.copy(), cv.COLOR_BGR2RGB)
            ax_path.imshow(img, origin='upper')
            # overlay path in pixel coordinates
            pix_pts = np.array([hf.mR2pix(p) for p in path])
            ax_path.plot(pix_pts[:, 0], pix_pts[:, 1], linewidth=2, label='path (pixels)')
            # plot route_list waypoints
            route_pix = np.array([hf.mR2pix(np.array([x, y])) for (x, y, _) in self.route_list])
            if len(route_pix) > 0:
                ax_path.scatter(route_pix[:, 0], route_pix[:, 1], c='red', s=20, label='route nodes')
            ax_path.set_title("Path over Map (pixels)")
            ax_path.axis('off')
        else:
            ax_path.plot(path[:, 0], path[:, 1], linewidth=2, label='path (m)')
            # plot route_list waypoints in metric coords if available
            if len(self.route_list) > 0:
                route_xy = np.array([[x, y] for (x, y, _) in self.route_list])
                ax_path.scatter(route_xy[:, 0], route_xy[:, 1], c='red', s=20, label='route nodes')
            ax_path.set_aspect('equal', 'box')
            ax_path.set_xlabel('x [m]')
            ax_path.set_ylabel('y [m]')
            ax_path.set_title('Path (XY)')
            ax_path.grid(True)

        ax_path.legend(loc='best')

        # --- Curvature vs arc-length ---
        ax_k = fig.add_subplot(gs[1, 0], sharex=None)
        # curvature array might be shorter than s if endpoints duplicated; align lengths:
        if len(k) != len(s):
            # if k has fewer points, interpolate to s
            k_s = np.interp(s, np.linspace(s[0], s[-1], len(k)), k)
        else:
            k_s = k

        ax_k.plot(s, k_s, linewidth=1.5, label='curvature κ(s)')
        ax_k.set_xlabel('arc-length s [m]')
        ax_k.set_ylabel('curvature [1/m]')
        ax_k.set_title('Curvature vs Arc-length')
        ax_k.grid(True)
        ax_k.legend(loc='best')

        plt.show()


if __name__ == "__main__":
    
    import cv2 as cv

    map_img = cv.imread('data/2024_VerySmall.png')

    planner = PathPlanning(map_img, show_imgs=True)

    nodes_to_pass = [73, 97, 100, 130, 140]
    planner.generate_path_passing_through(nodes_to_pass, step_length=0.01, method='spline')

    # Draw nodes and display path on the OpenCV map window (keeps existing behavior)
    planner.draw_path()

    # Plot matplotlib figures: path (XY) and curvature vs arc-length.
    # set show_map=True to overlay on the map image (pixel coords).
    planner.plot_path_and_curvature(show_map=True)

    cv.waitKey(0)
    cv.destroyAllWindows()