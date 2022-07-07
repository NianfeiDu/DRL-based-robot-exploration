import datetime
import math
import os
import pprint
import random
import time
import traceback
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


def draw(width, height, map, front_cells):
    tem_map = np.zeros((width, height), dtype=int)
    for i in range(width):
        for j in range(height):
            if map[i, j] == -1:
                tem_map[i, j] = 127
            elif map[i, j] == 0:
                tem_map[i, j] = 255
            elif map[i, j] == 100:
                tem_map[i, j] = 0
            else:
                tem_map[i, j] = map[i, j]
    plt.figure()
    ax = plt.subplot()
    ax.imshow(tem_map, cmap="gray")
    # for i, j in front_cells:
    #     tem_map[i, j] = int(255 * 0.75)
    f = np.array(front_cells)
    ax.scatter(f[:, 1], f[:, 0])
    plt.show()
    plt.axis("off")
    return tem_map


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
    map_msg = mapmsg
    resolution = map_msg.info.resolution
    origin_x = map_msg.info.origin.position.x
    origin_y = map_msg.info.origin.position.y
    data = map_msg.data
    # grid = np.array(data).reshape(map_msg.info.height, map_msg.info.width)
    grid = np.array(data)
    size = grid.size
    sampled_grid = grid[0 : size - 1 : size // 8100][3:8100]
    # next_state = sampled_grid
    next_state = np.concatenate((sampled_grid, np.array(self.pose_map)))

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
    frontier_cells_index2 = []
    unkonow_area = 0
    grid_frontier_i = np.zeros(size, dtype=np.uint8)
    for i in range(1, mapmsg.info.height - 1):
        for j in range(1, mapmsg.info.width - 1):
            if grid[i, j] == 0:
                near = grid[i - 1 : i + 2, j - 1 : j + 2].reshape(1, -1)[0]
                if near.__contains__(-1):
                    frontier_cells_index2.append([i, j])
                cnt = 0
                for near_i in list(near):
                    cnt = cnt + 1 if near_i == -1 else cnt
                    if cnt >= 3 and i >= 3 and j >= 3:
                        grid_near = grid[i - 3 : i + 4, j - 3 : j + 4].reshape(1, -1)[0]
                        if not grid_near.__contains__(100):
                            frontier_cells_index.append([i, j])
                            grid_frontier_i[i, j] = 1

            if grid[i, j] == -1:
                unkonow_area += 1
                # if near.__contains__(-1):
                #     frontier_cells_index.append([i, j])
                #     grid_frontier_i[i, j] = 1

    print(unkonow_area, height, width)
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
    # plt.figure()
    # ax = plt.subplot()
    # ax.imshow(tem_map, cmap="gray")
    # # f = np.array(frontier_cells)
    # # ax.scatter(f[:, 1], f[:, 0], c="g")
    # fi = np.array(frontier_cells_index)
    # ax.scatter(fi[:, 1], fi[:, 0])
    # plt.show()
    # plt.axis("off")
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
    frontier_cells_index2_map = []
    for ele in frontier_cells_index2:
        frontier_cells_index2_map.append(
            [ele[1] * resolution + origin_y, ele[0] * resolution + origin_x]
        )
    point_pub_i = rospy.Publisher("/fronts_index", PointStamped, queue_size=1)
    point_pub = rospy.Publisher("/fronts", PointStamped, queue_size=1)
    cen_pub = rospy.Publisher("/centers", PointStamped, queue_size=1)
    for _ in range(5):
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

        # for i, cell in enumerate(frontier_cells_index2_map):
        #     point = PointStamped()
        #     point.header.seq = i
        #     point.header.frame_id = "map"
        #     point.header.stamp = rospy.Time.now()
        #     point.point.x = cell[0]
        #     point.point.y = cell[1]
        #     point.point.z = 0
        #     point_pub.publish(point)
        #     rospy.sleep(0.001)
        # for i, cell in enumerate(cents_sorted_map):
        #     point = PointStamped()
        #     point.header.seq = i
        #     point.header.frame_id = "map"
        #     point.header.stamp = rospy.Time.now()
        #     point.point.x = cell[0]
        #     point.point.y = cell[1]
        #     point.point.z = 0
        #     cen_pub.publish(point)
        #     rospy.sleep(0.001)
        rospy.spin()
    # 0 * resolution + origin_x, 0 * resolution + origin_y
    # client = actionlib.SimpleActionClient("move_base", move_base_msgs.MoveBaseAction)
    # client.wait_for_server()
    # rospy.logwarn("end waiting for action move_base")

    # goalB = Pose()  # goalB is next frontier
    # goalB.position.x = 0 * resolution + origin_x
    # goalB.position.y = 0 * resolution + origin_y
    # # goalB.orientation.z = self.odom[2]
    # goalB.orientation.z = random.uniform(0, 1)
    # # goalB.orientation.w = self.odom[3]
    # goalB.orientation.w = math.sqrt(1 - goalB.orientation.z**2)

    # goal = move_base_msgs.MoveBaseGoal()
    # goal.target_pose.pose = goalB
    # goal.target_pose.header.stamp = rospy.Time.now()
    # goal.target_pose.header.frame_id = "map"  # odom,  map
    # # Send the goal!
    # # print"sending goal"
    # client.send_goal(goal)
    # while client.get_state() != 3:
    #     print(str(client.get_state()))
    #     rospy.sleep(0.2)


from nav_msgs.srv import GetPlan

if __name__ == "__main__":

    # main()
    env = robot_rl_env.RobotEnv()
    env.reset()
    while env.done is False:
        env.nf_nav2(0)
        rospy.sleep(0.2)
    # dt_now = datetime.datetime.now()
    # fpath = "/home/nianfei/ros_ws/src/dqn_for_slam/dqn_for_slam/log/maze7_trace/nf_trace_{}_{}{}{}.txt".format(
    #     env.total_step, dt_now.month, dt_now.day, dt_now.hour
    # )
    # with open(fpath, "w+") as f:
    #     f.write(pprint.pformat(env.full_trace))

    """ "move base test"""
    # make_plan = rospy.ServiceProxy("/move_base/make_plan", GetPlan)
    # env = robot_rl_env.RobotEnv()
    # goal = env.setupGoals(2, -2)
    # plan_start, plan_goal, plan_tolerance = env.set_plan_goal((2, -2))
    # # get trajectory using servise move_base
    # plan_response = make_plan(plan_start, plan_goal, plan_tolerance)
    # env.navigateToGoal(goal)
    # trace_list = []
    # plan_x = np.empty((len(plan_response.plan.poses), 1))
    # plan_y = np.empty((len(plan_response.plan.poses), 1))
    # i = 0
    # for plan_pose in plan_response.plan.poses:
    #     plan_x[i] = plan_pose.pose.position.x  # x
    #     plan_y[i] = plan_pose.pose.position.y  # y
    #     trace_list.append((plan_x[i][0], plan_x[i][0]))
    #     # plan_w[i] = plan_pose.pose.orientation.w  #w
    #     i = i + 1
    # # trace_list = trace_list[::5, :]
    # pprint(trace_list)
    # env.navigateToGoal(goal)
    # print(env.observation_space.shape)
    # for _ in range(50):
    #     act = env.action_space.sample()
    #     env.step(act)
    # rospy.init_node("tesp")

    """
    # a step for env
    while True:
        cen = None
        # wait for valid act and center
        while cen is None:
            act = env.action_space.sample()
            env.process()
            cen = env.valid_act(act)
            print("cents_sorted_world")
            pprint(env.cents_sorted_world)

            pprint(env.valid_act_list(), width=10)
        print("cen", cen, "odom", env.odom, "act", act)
        try:
            goal = env.setupGoals(cen[0], cen[1])
            rospy.logwarn("go to: {}".format(goal))
            env.navigateToGoal(goal)
            manh_list = []
            while str(env.client.get_state()) != "3":
                env.callback_odom()
                manh = abs(env.odom[0] - cen[0]) + abs(env.odom[1] - cen[1])
                if len(manh_list) < 100:
                    manh_list.append(manh)
                else:
                    manh_list.pop(0)
                    manh_list.append(manh)
                    if sum(manh_list) / 100 < 0.5:
                        print("almost move to the goal ", cen)
                        break
                    if sum(manh_list) / 100 > 0.5 and np.std(np.array(manh_list)) < 4:
                        print("stuck at ", cen)
                        break
                rospy.sleep(0.02)

            env.resetCostmaps()
        except:
            print(traceback.print_exc())"""
