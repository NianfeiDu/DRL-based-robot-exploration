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

import numpy as np
import rospy
from cv2 import sqrt
from geometry_msgs.msg import PointStamped
from gym import spaces
from nav_msgs.msg import MapMetaData, OccupancyGrid, Odometry
from nav_msgs.srv import GetPlan
from robot_localization.srv import SetPose
from sensor_msgs.msg import LaserScan
from skimage import measure
from skimage.measure import label, regionprops
from std_srvs.srv import Empty

import robot_rl_env


def callback_map(OccupancyGrid):
    """update self.rawmap and self.p0(info.origin\info.resolution)
    Args:
        OccupancyGrid (_type_): _description_
    """
    mapmsg = OccupancyGrid
    map = np.array(mapmsg.data, dtype=np.int32)
    width = int(mapmsg.info.width)
    height = int(mapmsg.info.height)
    size = map.size
    grid = map.reshape(width, height)
    # frontier detection
    frontier_cells_index = []
    unkonow_area = 0
    for i in range(1, mapmsg.info.height - 1):
        for j in range(1, mapmsg.info.width - 1):
            if grid[i, j] == 0:
                near = grid[i - 1 : i + 2, j - 1 : j + 2].reshape(1, -1)[0]
                cnt = 0
                for near_i in list(near):
                    cnt = cnt + 1 if near_i == -1 else cnt
                    if cnt >= 3 and i >= 3 and j >= 3:
                        grid_near = grid[i - 3 : i + 4, j - 3 : j + 4].reshape(1, -1)[0]
                        if not grid_near.__contains__(100):
                            frontier_cells_index.append([i, j])
            if grid[i, j] == -1:
                unkonow_area += 1
    resolution = mapmsg.info.resolution
    origin_x = mapmsg.info.origin.position.x
    origin_y = mapmsg.info.origin.position.y
    data = mapmsg.data
    grid = np.array(data).reshape(mapmsg.info.height, mapmsg.info.width)
    size = grid.shape

    frontier_cells_index_map = []
    for ele in frontier_cells_index:
        frontier_cells_index_map.append(
            [ele[1] * resolution + origin_y, ele[0] * resolution + origin_x]
        )
    point_pub_i = rospy.Publisher("/fronts_index", PointStamped, queue_size=1)
    for i, cell in enumerate(frontier_cells_index_map):
        point = PointStamped()
        point.header.seq = i
        point.header.frame_id = "map"
        point.header.stamp = rospy.Time.now()
        point.point.x = cell[0]
        point.point.y = cell[1]
        point.point.z = 0
        point_pub_i.publish(point)


def main():
    rospy.init_node("front_pub", anonymous=True)
    sub_map = rospy.Subscriber("/map", OccupancyGrid, callback_map)
    rospy.spin()


if __name__ == "__main__":
    main()
