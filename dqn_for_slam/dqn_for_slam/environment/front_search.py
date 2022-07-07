import math
import os
import pprint
import random
import time
import traceback
from pprint import pprint
from time import time
from tkinter.messagebox import NO
from turtle import distance
from typing import List, Tuple

import actionlib
import cv2
import gym
import matplotlib.pyplot as plt
import move_base_msgs.msg as move_base_msgs
import numpy as np
import PIL
import rosparam
import rospy
import std_msgs.msg as std_msgs
import std_srvs.srv as std_srvs
import tf
import tf2_ros
from actionlib_msgs.msg import GoalStatus
from cv2 import sqrt
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import (
    Point,
    PointStamped,
    Pose,
    PoseStamped,
    PoseWithCovarianceStamped,
    Quaternion,
    TransformStamped,
    Twist,
    Vector3,
)
from gym import spaces
from nav_msgs.msg import MapMetaData, OccupancyGrid, Odometry
from nav_msgs.srv import GetPlan
from robot_localization.srv import SetPose
from sensor_msgs.msg import LaserScan
from skimage import measure
from skimage.measure import label, regionprops
from std_srvs.srv import Empty

import robot_rl_env


def main():
    rospy.init_node("map", anonymous=True)
    rospy.logwarn("node start.")
    mapmsg = rospy.wait_for_message("/map", OccupancyGrid, None)
    map = np.array(mapmsg.data, dtype=np.int32)
    width = int(mapmsg.info.width)
    height = int(mapmsg.info.height)
    rospy.logwarn("received gridmap")
    size = map.size
    sam = map[0 : size - 1 : size // 100 + 1]
    grid = map.reshape(width, height)

    # --------------------------------------------- open cells ---------------------
    thresh_low = 0
    thresh_high = 10
    # "1" in dst_open opencell are the open cells
    # "0" in dst_open opencell are the unvisited cells and occupied cells
    opencell = grid == -1
    # detect contours
    contours_open = measure.find_contours(opencell, 0.5)
    contours_open_cell = list()
    for ele in contours_open:
        for cell in ele:
            contours_open_cell.append(cell.tolist())
    # --------------------------------------------- unvisited cells ---------------------
    thresh_low = -1
    thresh_high = -1
    unknown = ((grid <= thresh_high) & (grid >= thresh_low)) * 1.0  # threshold
    unkonow_area = np.sum(unknown)
    pprint(unknown)
    print(unkonow_area)
    # "1" in unknown are the unvisited cells
    # "0" in unknown are the open cells and occupied cells
    # detect contours
    contours_unvisited = measure.find_contours(unknown, 0.5)
    contours_unvisited_cell = list()

    for ele in contours_unvisited:
        for cell in ele:
            contours_unvisited_cell.append(cell.tolist())
    # -----------------occupied cells---------------------
    thresh_low = 90
    thresh_high = 100
    occup = ((grid <= thresh_high) & (grid >= thresh_low)) * 1.0
    # ----------------------------------------------------------------
    # frontier detection
    frontier_cells = [x for x in contours_unvisited_cell if x in contours_open_cell]
    size = grid.shape
    grid_frontier = np.zeros(size)
    for ele in frontier_cells:
        # grid_frontier[math.floor(ele[0]), math.floor(ele[1])] = 1
        grid_frontier[int(ele[0]), int(ele[1])] = 1

    frontier_cells_index = []
    unkonow_area = 0
    grid_frontier_i = np.zeros(size, dtype=np.uint8)
    for i in range(1, mapmsg.info.height - 1):
        for j in range(1, mapmsg.info.width - 1):
            if grid[i, j] == 0:
                near = grid[i - 1 : i + 2, j - 1 : j + 2].reshape(1, -1)[0]
                if near.__contains__(-1):
                    frontier_cells_index.append([i, j])
                    grid_frontier_i[i, j] = 1

    tem_map = np.zeros((width, height), dtype=int)
    for i in range(width):
        for j in range(height):
            if grid[i, j] == -1:
                tem_map[i, j] = 127
            elif grid[i, j] == 0:
                tem_map[i, j] = 255
            elif grid[i, j] == 100:
                tem_map[i, j] = 0
            else:
                tem_map[i, j] = map[i, j]

    resolution = mapmsg.info.resolution
    origin_x = mapmsg.info.origin.position.x
    origin_y = mapmsg.info.origin.position.y
    data = mapmsg.data
    grid = np.array(data).reshape(mapmsg.info.height, mapmsg.info.width)
    size = grid.shape

    robot_pose_world = [1.5, 2.5]  # robot initial position in world
    robot_pose_pixel = [0, 0]  # robot initial position in grid (pixel in image)
    robot_pose_pixel[0] = (robot_pose_world[0] - origin_x) / resolution
    robot_pose_pixel[1] = (robot_pose_world[1] - origin_y) / resolution
    # group them!
    conected_frontier, _ = measure.label(grid_frontier_i, return_num=True)
    manh_dist = []  # stores distances
    cents = []  # stores centers of frontiers
    for region in regionprops(conected_frontier):
        # take regions with large enough areas
        # FIXME:change region threshod
        if region.area >= 5:  # 30 do not consider small frontier groups
            # the centroid of each valid region
            cen_y = region.centroid[1]  # Centroid coordinate tuple (row, col)
            cen_x = region.centroid[0]  # Centroid coordinate tuple (row, col)
            cents.append([cen_x, cen_y])  # cents[col,row]
            # Manhattan Distance from robot to each frontier center
            manh = abs(cen_x - robot_pose_pixel[0]) + abs(cen_y - robot_pose_pixel[1])
            manh_dist.append(manh)
    # sort two list: centers of each frontiers according to the man_distance
    # a sorted record of all the candidate goals (close-far)
    plt.figure()
    ax = plt.subplot()
    ax.imshow(tem_map, cmap="gray")
    # f = np.array(frontier_cells)
    # ax.scatter(f[:, 1], f[:, 0], c="g")
    fi = np.array(frontier_cells_index)
    ax.scatter(fi[:, 1], fi[:, 0])
    plt.show()
    plt.axis("off")
    cents_sorted = [x for _, x in sorted(zip(manh_dist, cents))]
    cents_sorted_map = []
    for ele in cents_sorted:
        cents_sorted_map.append(
            [ele[0] * resolution + origin_x, ele[1] * resolution + origin_y]
        )
    frontier_cells_map = []
    for ele in frontier_cells:
        frontier_cells_map.append(
            [ele[1] * resolution + origin_y, ele[0] * resolution + origin_x]
        )
    frontier_cells_index_map = []
    for ele in frontier_cells_index:
        frontier_cells_index_map.append(
            [ele[1] * resolution + origin_y, ele[0] * resolution + origin_x]
        )
    point_pub_i = rospy.Publisher("/fronts_index", PointStamped, queue_size=1)
    point_pub = rospy.Publisher("/fronts", PointStamped, queue_size=1)
    cen_pub = rospy.Publisher("/centers", PointStamped, queue_size=1)
    while not rospy.is_shutdown():
        for i, cell in enumerate(frontier_cells_index_map):
            point = PointStamped()
            point.header.seq = i
            point.header.frame_id = "map"
            point.header.stamp = rospy.Time.now()
            point.point.x = cell[0]
            point.point.y = cell[1]
            point.point.z = 0
            point_pub_i.publish(point)
            rospy.sleep(0.01)


if __name__ == "main":
    main()
