#!/usr/bin/python3
import networkx as nx
import numpy as np
import cv2 as cv
from pyclothoids import Clothoid
import helper_functions as hf
import os

YAW_DIFF_THRESHOLD = 90  # [deg] threshold on difference between yaw from car and edge in 'get_closest_node'
SHOW_IMGS = False


class PathPlanning:
    def __init__(self, map_img):
        # start and end nodes
        self.source = str(472)  # BFMC_2024
        self.target = str(468)  # BFMC_2024

        # initialize path
        self.path = []
        self.navigator = []   # set of instruction for the navigator
        self.path_data = []   # additional data for the path (e.g. curvature)
        self.step_length = 0.01

        # previous index of the closest point on the path to the vehicle
        self.prev_index = 0

        base_dir = os.path.dirname(__file__)
        graph_path = os.path.join(base_dir, 'data', 'final_graph.graphml')
        self.G = nx.read_graphml(graph_path)

        # initialize route subgraph and list for interpolation
        self.route_graph = nx.DiGraph()
        self.route_list = []
        self.old_nearest_point_index = None  # used in search target point

        self.nodes_data = self.G.nodes.data()
        self.edges_data = self.G.edges.data()

        # load intersection files
        intersection_cen_path = os.path.join(base_dir, 'data', 'int_mid.txt')
        intersection_in_path = os.path.join(base_dir, 'data', 'int_in.txt')
        intersection_out_path = os.path.join(base_dir, 'data', 'int_out.txt')

        self.intersection_cen = list(np.loadtxt(intersection_cen_path, dtype=str))
        self.intersection_in = list(np.loadtxt(intersection_in_path, dtype=str))
        self.intersection_out = list(np.loadtxt(intersection_out_path, dtype=str))

        # Skip nodes and forbidden nodes
        self.skip_nodes = [str(i) for i in [262, 235, 195, 196, 281, 216, 263, 234, 239, 301, 282, 258]]
        self.no_yaw_calibration_nodes = [str(i) for i in [161, 162, 163, 164, 165]]  # for compatibility
        self.forbidden_nodes = self.intersection_cen + self.intersection_in + self.intersection_out

        # import nodes and edges
        self.list_of_nodes = list(self.G.nodes)
        self.list_of_edges = list(self.G.edges)

        # possible starting positions
        all_start_nodes = list(self.G.nodes)
        self.all_start_nodes = []
        for n in all_start_nodes:
            if n in self.forbidden_nodes:
                continue
            else:
                self.all_start_nodes.append(n)
        self.all_nodes_coords = np.array([self.get_coord(node) for node in self.all_start_nodes])

        self.bumpy_road_nodes = ['0']  # BFMC_2024
        self.junctions = ['0']         # BFMC_2024

        # map to plot trajectory and car
        self.map = map_img

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

        # remove duplicates
        prev_x, prev_y = 0, 0
        for i, (x, y, yaw) in enumerate(self.route_list):
            if np.linalg.norm(np.array([x, y]) - np.array([prev_x, prev_y])) < 0.001:
                self.route_list.pop(i)
            prev_x, prev_y = x, y

        # interpolate
        self.path = PathPlanning.interpolate_route(self.route_list, step_length)
        return self.path

    def generate_path_passing_through(self, list_of_nodes, step_length=0.05):
        """Extend the path generation from source-target to a sequence of nodes/locations"""
        assert len(list_of_nodes) >= 2, "List of nodes must have at least 2 nodes"
        print("Generating path passing through:", list_of_nodes)

        src, tgt = list_of_nodes[0], list_of_nodes[1]
        complete_path = self.compute_shortest_path(source=src, target=tgt, step_length=step_length)

        for i in range(1, len(list_of_nodes) - 1):
            src, tgt = list_of_nodes[i], list_of_nodes[i+1]
            self.compute_shortest_path(source=src, target=tgt, step_length=step_length)
            self.path = self.path[1:]
            complete_path = np.concatenate((complete_path, self.path))

        self.path = complete_path

    def get_path_ahead(self, index, look_ahead=100):
        assert 0 <= index < len(self.path)
        return np.array(self.path[index:min(index + look_ahead, len(self.path)-1), :])

    def print_path_info(self):
        prev_state, prev_next_state = None, None
        for i in range(len(self.path_data) - 1):
            curr_state, next_state = self.path_data[i][0], self.path_data[i][1]
            if curr_state != prev_state or next_state != prev_next_state:
                print(f'{i}: {self.path_data[i]}')
            prev_state, prev_next_state = curr_state, next_state

    def get_length(self, path=None):
        """Calculates the length of the trajectory"""
        if path is None:
            return 0
        length = 0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            length += np.hypot(x2 - x1, y2 - y1)
        return length

    def get_coord(self, node):
        x = self.nodes_data[node]['x']
        y = self.nodes_data[node]['y']
        p = np.array([x, y])
        return hf.mL2mR(p)

    def get_path(self):
        return self.path

    def print_navigation_instructions(self):
        for i, instruction in enumerate(self.navigator):
            print(i+1, ")", instruction)

    @staticmethod
    def compute_path(xi, yi, thi, xf, yf, thf, step_length):
        clothoid_path = Clothoid.G1Hermite(xi, yi, thi, xf, yf, thf)
        length = clothoid_path.length
        X, Y = clothoid_path.SampleXY(int(length / step_length))
        return [X, Y]

    @staticmethod
    def interpolate_route(route, step_length):
        path_X, path_Y = [], []

        for i in range(len(route) - 1):
            xc, yc, thc = route[i]
            xn, yn, thn = route[i+1]
            thc, thn = np.deg2rad(thc), np.deg2rad(thn)

            X, Y = PathPlanning.compute_path(xc, yc, thc, xn, yn, thn, step_length)
            path_X.extend(X)
            path_Y.extend(Y)

        # build array for cv.polylines
        path, prev_x, prev_y = [], 0.0, 0.0
        for x, y in zip(path_X, path_Y):
            if not (np.isclose(x, prev_x, rtol=1e-5) and np.isclose(y, prev_y, rtol=1e-5)):
                path.append([x, y])
            prev_x, prev_y = x, y

        return np.array(path, dtype=np.float32)

    def draw_path(self):
        # draw nodes
        for node in self.list_of_nodes:
            p = self.get_coord(node)
            cv.circle(self.map, hf.mR2pix(p), 5, (0, 0, 255), -1)
            cv.putText(self.map, str(node), hf.mR2pix(p), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # draw all points in given path
        for point in self.route_list:
            x, y, _ = point
            p = np.array([x, y])
            cv.circle(self.map, hf.mR2pix(p), 5, (255, 0, 0), 1)

        # draw trajectory
        cv.polylines(self.map, [hf.mR2pix(self.path)], False, (200, 200, 0), thickness=4, lineType=cv.LINE_AA)
        if SHOW_IMGS:
            cv.imshow('Path', self.map)

    def get_closest_node(self, p):
        """Returns the closest node to the given point np.array([x,y])"""
        diff = self.all_nodes_coords - p
        dist = np.linalg.norm(diff, axis=1)
        index_closest = np.argmin(dist)
        return self.all_start_nodes[index_closest], dist[index_closest]

    def get_closest_node_start(self, p, car_yaw):
        """Returns the closest valid start node to the given point with orientation check"""
        car_yaw = np.deg2rad((car_yaw + 180) % 360 - 180)

        diff = self.all_nodes_coords - p
        dist = np.linalg.norm(diff, axis=1)
        ordered_list = sorted(enumerate(dist), key=lambda x: x[1])

        for idx, d in ordered_list:
            successor_list = list(self.G.successors(self.all_start_nodes[idx]))
            successor_node = successor_list[0]

            closest_node_coords = self.get_coord(self.all_start_nodes[idx])
            successor_coords = self.get_coord(successor_node)

            edge_yaw = np.arctan2(successor_coords[1] - closest_node_coords[1],
                                  successor_coords[0] - closest_node_coords[0])
            error_yaw = edge_yaw - car_yaw

            if error_yaw < np.deg2rad(YAW_DIFF_THRESHOLD):
                print("Closest node is", self.all_start_nodes[idx])
                return self.all_start_nodes[idx], d

        print("Error: impossible to find closest node")