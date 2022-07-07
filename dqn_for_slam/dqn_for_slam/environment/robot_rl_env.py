import datetime
import math
import os
import pprint
import random
import time
import traceback
from tkinter.messagebox import NO
from turtle import distance
from typing import List, Tuple

import actionlib
import gym
import move_base_msgs.msg as move_base_msgs
import numpy as np
import rosparam
import rospy
import std_msgs.msg as std_msgs
import std_srvs.srv as std_srvs
import tf
import tf2_ros
# from actionlib_msgs.msg import GoalStatus
# from cv2 import sqrt
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import (Point, Pose, PoseStamped,
                               PoseWithCovarianceStamped, Quaternion,
                               TransformStamped, Twist, Vector3)
from gym import spaces
from nav_msgs.msg import MapMetaData, OccupancyGrid, Odometry
from nav_msgs.srv import GetPlan
# from nav_msgs.srv import GetPlan
from robot_localization.srv import SetPose
from sensor_msgs.msg import LaserScan
from skimage import measure
from skimage.measure import label, regionprops
from std_srvs.srv import Empty

file_path = __file__
dir_path = (
    file_path[: (len(file_path) - len("environment/robot_rl_env.py"))] + "config/"
)
config_file_name = "rlslam_map_reward.yaml"
config_file_path = os.path.join(dir_path, config_file_name)
parameters_list = rosparam.load_file(config_file_path)
for params, namespace in parameters_list:
    rosparam.upload_params(namespace, params)

INITIAL_POS_X = rospy.get_param("rlslam/initial_posx")
INITIAL_POS_Y = rospy.get_param("rlslam/initial_posy")

LIDAR_SCAN_MAX_DISTANCE = rospy.get_param("rlslam/scan_max_distance")
LIDAR_SCAN_MIN_DISTANCE = rospy.get_param("rlslam/scan_min_distance")
TRAINING_IMAGE_SIZE = rospy.get_param("rlslam/training_image_size")
MAZE_SIZE = rospy.get_param("rlslam/maze_size")
MAP_COMPLETENESS_THRESHOLD = rospy.get_param("rlslam/map_completed_threshold")
COLLISION_THRESHOLD = rospy.get_param("rlslam/crash_distance")

REWARD_MAP_COMPLETED = rospy.get_param("rlslam/reward_map_completed")
REWARD_CRASHED = rospy.get_param("rlslam/reward_crashed")

MAX_PX = rospy.get_param("rlslam/obs_space_max/px")
MAX_PY = rospy.get_param("rlslam/obs_space_max/py")
MAX_QZ = rospy.get_param("rlslam/obs_space_max/qz")
MAX_ACTION_NUM = 3
MAX_MAP_COMPLETENESS = 100.0
MAX_STEPS = rospy.get_param("rlslam/steps_in_episode")

MIN_PX = rospy.get_param("rlslam/obs_space_min/px")
MIN_PY = rospy.get_param("rlslam/obs_space_min/py")
MIN_QZ = rospy.get_param("rlslam/obs_space_min/qz")
MIN_ACTION_NUM = -1
MIN_STEPS = 0
MIN_MAP_COMPLETENESS = 0.0

MAP_SIZE = (MAX_PX - MIN_PX) * (MAX_PY - MIN_PY)

STEERING = 2 * rospy.get_param("rlslam/steering")
THROTTLE = rospy.get_param("rlslam/throttle")

TIMEOUT = rospy.get_param("rlslam/timeout")
SLEEP_RESET_TIME = rospy.get_param("rlslam/sleep_reset_time")


class RobotEnv(gym.Env):
    """
    Environment for reinforce learning
    """

    def __init__(self) -> None:
        rospy.init_node("rl_dqn", anonymous=True)
        self.counter_odom = 0
        self.position = Point(INITIAL_POS_X, INITIAL_POS_Y, 0)
        self.orientation = Quaternion(1, 0, 0, 0)
        self.ranges = None
        self.map_completeness_pct = MIN_MAP_COMPLETENESS
        self.occupancy_grid = None
        self.done = False
        self.steps_in_episode = 0
        self.min_distance = 100
        self.reward = None
        self.reward_in_episode = 0
        self.now_action = -1
        self.last_action = -1
        self.last_map_completeness_pct = 0
        self.map_size_ratio = MAZE_SIZE / MAP_SIZE
        self.action_space = spaces.Discrete(4)

        # define observation space
        map_msg = None
        while map_msg is None:
            try:
                map_msg = rospy.wait_for_message("/map", OccupancyGrid, 3)
            except:
                rospy.sleep(0.1)
        grid = np.array(map_msg.data)
        size = grid.size
        sampled_grid = grid[0 : size - 1 : size // 8100][3:8100]
        stete_shape = sampled_grid.size
        map_low = np.array([-1] * stete_shape)
        map_low = np.concatenate((map_low, np.array([-20, -20, -4])))

        map_high = np.array([100] * stete_shape)
        map_high = np.concatenate((map_high, np.array([20, 20, 4])))
        self.observation_space = spaces.Box(map_low, map_high, dtype=np.float32)
        # self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # ROS initialization
        self.ack_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.map_reset_service = rospy.ServiceProxy("/clear_map", Empty)
        self.gazebo_model_state_service = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState
        )
        # self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        # self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_odom_to_base = rospy.ServiceProxy("/set_pose", SetPose)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.distance = lambda p1, p2: math.sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
        )

        # may not used
        self.clear_costmap_srv = None
        # self.pub = rospy.Publisher(
        #     "/mobile_base/commands/reset_odometry", std_msgs.Empty, queue_size=1
        # )
        self.tflistener = tf.TransformListener()
        self.client = actionlib.SimpleActionClient(
            "move_base", move_base_msgs.MoveBaseAction
        )
        self.sub_map = rospy.Subscriber("/map", OccupancyGrid, self.callback_map)
        self.sub_odom = rospy.Subscriber(
            "/steer_drive_controller/odom", Odometry, self.callback_odom2
        )

        # self.make_plan = rospy.ServiceProxy("/move_base/make_plan", GetPlan)
        self.client.wait_for_server()
        # rospy.logwarn("end waiting for action move_base")
        self.stop = False
        self.flag_stuck = 0
        self.pos_list = []
        self.goal = []
        self.cents_sorted_world = []
        self.frontier_cells = []
        self.trace_instep = []
        self.unkonow_area = 0
        self.odom_dist = []
        self.stuck_pose = []
        self.total_step = 0
        self.occmap = None
        self.odom = None
        self.full_trace = []

    def callback_map(self, OccupancyGrid):
        """update self.rawmap and self.p0(info.origin\info.resolution)
        Args:
            OccupancyGrid (_type_): _description_
        """
        self.occmap = OccupancyGrid

    def reset(self) -> np.ndarray:
        """
        initiate status and  return the first observed values
        """
        rospy.loginfo("start resetting")

        self.done = False
        self.position = Point(INITIAL_POS_X, INITIAL_POS_Y, 0)
        self.orientation = Quaternion(1, 0, 0, 0)
        self.steps_in_episode = 0
        self.map_completeness_pct = 0
        self.last_map_completeness_pct = 0
        self.reward_in_episode = 0
        self.occupancy_grid = None
        self.ranges = None
        self._reset_rosbot()
        self._reset_tf()
        self._send_action(0, 0)
        self.client.wait_for_server()
        rospy.logwarn("end waiting for action move_base")
        self.stop = False
        self.flag_stuck = 0
        self.pos_list = []
        self.goal = []
        self.cents_sorted_world = []
        self.frontier_cells = []
        self.trace_instep = []
        self.unkonow_area = 0
        self.odom_dist = []
        self.stuck_pose = []
        self.odom = None
        self.full_trace = []

        # clear map
        rospy.wait_for_service("/clear_map")
        if self.map_reset_service():
            rospy.loginfo("reset map")
        else:
            rospy.logerror("could not reset map")

        self._update_map_size_ratio()  # sometimes map expands
        """wait for gridmap"""
        while self.occmap is None:
            try:
                self.occmap = rospy.wait_for_message("/map", OccupancyGrid, 3)
            except:
                rospy.sleep(0.1)

        self.process()
        while len(self.frontier_cells) < 10:
            rospy.loginfo("reset map beceuse no frontier_cells found after reset")
            self.occmap = rospy.wait_for_message("/map", OccupancyGrid, 3)
            self.process()
            rospy.sleep(1)
        self.reward = 0
        next_state = self.get_state()
        return next_state

    def _reset_tf(self) -> None:
        rospy.wait_for_service("/set_pose", 5)
        tf_pose = PoseWithCovarianceStamped()
        tf_pose.pose.pose.position.x = self.position.x
        tf_pose.pose.pose.position.y = self.position.y
        tf_pose.pose.pose.orientation.w = self.orientation.w
        if self.reset_odom_to_base(tf_pose):
            rospy.loginfo("initialized tf")
        else:
            rospy.logerror("/set_pose service call failed")

    def _reset_rosbot(self) -> None:
        """ """
        rospy.wait_for_service("gazebo/set_model_state", 5)

        model_state = ModelState()
        model_state.model_name = "rosbot"
        model_state.pose.position.x = INITIAL_POS_X
        model_state.pose.position.y = INITIAL_POS_Y
        model_state.pose.position.z = 0
        model_state.pose.orientation.x = 0
        model_state.pose.orientation.y = 0
        model_state.pose.orientation.z = 0
        model_state.pose.orientation.w = 1
        model_state.twist.linear.x = 0
        model_state.twist.linear.y = 0
        model_state.twist.linear.z = 0
        model_state.twist.angular.x = 0
        model_state.twist.angular.y = 0
        model_state.twist.angular.z = 0

        if self.gazebo_model_state_service(model_state):
            rospy.loginfo("set robot init state")
        else:
            rospy.logerror("/gazebo/set_model_state service call failed")

    def _update_map_size_ratio(self) -> None:
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message(
                    "/map_metadata", MapMetaData, timeout=TIMEOUT
                )
            except:
                pass
        width = data.resolution * data.width
        height = data.resolution * data.height

        self.map_size_ratio = float(MAZE_SIZE / (width * height))

    def step_old(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # rospy.loginfo('start step' + str(self.steps_in_episode + 1))

        # rospy.wait_for_service('/gazebo/unpause_physics')
        # try:
        #    self.unpause()
        # except (rospy.ServiceException) as e:
        #    rospy.loginfo("/gazebo/unpause_physics service call failed")

        self.last_action = self.now_action
        self.now_action = action

        if self.reward == -5:
            if self.last_action == 3:
                action = 2
            elif self.last_action == 2:
                action = 3
            elif self.last_action == 1:
                action = 3
            elif self.last_action == 0:
                action = 3
        if action == 0:  # turn left
            steering = STEERING
            throttle = THROTTLE
        elif action == 1:  # turn right
            steering = -1 * STEERING
            throttle = THROTTLE
        elif action == 2:  # straight
            steering = 0
            throttle = THROTTLE
        elif action == 3:  # backward
            steering = 0
            throttle = -1 * THROTTLE
        else:
            raise ValueError("Invalid action")

        # initialize rewards, next_state, done
        self.reward = None
        self.done = False
        self.steps_in_episode += 1
        # FIXME: car up and down bug fix:sleep when action is straight or back
        if action == 3:
            if self.last_action != 3:
                self._send_action(0, 0)
                time.sleep(0.1)
            self._send_action(steering, throttle)
            time.sleep(0.1)
        else:
            if self.last_action == 3:
                self._send_action(0, 0)
                time.sleep(0.1)
            self._send_action(steering, throttle)
            time.sleep(0.1)

        if self.steps_in_episode % 50 == 0:
            self._send_action(0, 0)
            time.sleep(0.5)
            self._send_action(steering, throttle)

        # time.sleep(SLEEP_BETWEEN_ACTION_AND_REWARD_CALCULATION)
        self._wait_until_twist_achieved(steering, throttle)
        sensor_state = self._update_scan()
        self._update_map_completeness()
        numeric_state = self._update_odom()

        # if self.steps_in_episode >= MAX_STEPS:
        #  rospy.wait_for_service('/gazebo/pause_physics')
        #  try:
        #      self.pause()
        #  except (rospy.ServiceException) as e:
        #      rospy.loginfo("/gazebo/pause_physics service call failed")

        next_state = np.concatenate([sensor_state, numeric_state])
        self._infer_reward()

        info = {}
        return next_state, self.reward, self.done, info

    def _wait_until_twist_achieved(self, angular_speed, linear_speed):
        """
        We wait for the cmd_vel twist given to be reached by the robot reading
        Bare in mind that the angular wont be controled , because its too imprecise.
        We will only consider to check if its moving or not inside the angular_speed_noise fluctiations it has.
        from the odometry.
        :param cmd_vel_value: Twist we want to wait to reach.
        :return:
        """
        rospy.loginfo("START wait_until_twist_achieved...")

        epsilon = 0.05
        update_rate = 10
        angular_speed_noise = 0.005
        rate = rospy.Rate(update_rate)

        angular_speed_is = self._check_angular_speed_dir(
            angular_speed, angular_speed_noise
        )

        linear_speed_plus = linear_speed + epsilon
        linear_speed_minus = linear_speed - epsilon

        roop_count = 0
        while not rospy.is_shutdown():
            current_odometry = None
            while current_odometry is None and not rospy.is_shutdown():
                try:
                    current_odometry = rospy.wait_for_message(
                        "/steer_drive_controller/odom", Odometry, timeout=TIMEOUT
                    )

                except:
                    rospy.logerror(
                        "Current /odom not ready yet, retrying for getting odom"
                    )

            odom_linear_vel = current_odometry.twist.twist.linear.x
            odom_angular_vel = current_odometry.twist.twist.angular.z

            linear_vel_are_close = (odom_linear_vel <= linear_speed_plus) and (
                odom_linear_vel > linear_speed_minus
            )
            odom_angular_speed_is = self._check_angular_speed_dir(
                odom_angular_vel, angular_speed_noise
            )

            # We check if its turning in the same diretion or has stopped
            angular_vel_are_close = angular_speed_is == odom_angular_speed_is

            if linear_vel_are_close and angular_vel_are_close:
                rospy.loginfo("Reached Velocity!")
                break

            roop_count += 1
            if roop_count >= 3:
                rospy.logwarn("its regarded as crashed")
                break

            rospy.logwarn("Not there yet, keep waiting...")
            rate.sleep()

    def _check_angular_speed_dir(self, angular_speed, angular_speed_noise):
        """
        It States if the speed is zero, posititive or negative
        """
        # We check if odom angular speed is positive or negative or "zero"
        if -angular_speed_noise < angular_speed <= angular_speed_noise:
            angular_speed_is = 0
        elif angular_speed > angular_speed_noise:
            angular_speed_is = 1
        elif angular_speed <= angular_speed_noise:
            angular_speed_is = -1
        else:
            angular_speed_is = 0
            rospy.logerror("Angular Speed has wrong value==" + str(angular_speed))

    def _update_scan(self) -> None:
        """ """
        rospy.loginfo("waiting lidar scan")
        # adapt number of sensor information to TRAINING_IMAGE_SIZE
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message(
                    "/scan_filtered", LaserScan, timeout=TIMEOUT
                )
            except:
                pass

        sensor_state = []
        self.min_distance = LIDAR_SCAN_MAX_DISTANCE
        mod = len(data.ranges) / TRAINING_IMAGE_SIZE
        for i, item in enumerate(data.ranges):
            if i % mod == 0:
                if np.isnan(data.ranges[i]):
                    sensor_state.append(LIDAR_SCAN_MAX_DISTANCE + 1.0)
                else:
                    sensor_state.append(data.ranges[i])
            if self.min_distance > data.ranges[i]:
                self.min_distance = data.ranges[i]

        return sensor_state

    def _update_odom(self):
        # rospy.loginfo("waiting odom")
        trans = None
        while trans is None and not rospy.is_shutdown():
            try:
                # listen to transform
                trans = self.tfBuffer.lookup_transform(
                    "map", "base_link", rospy.Time(0)
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                pass
        # rospy.loginfo("end waiting odom")

        self.position.x = trans.transform.translation.x
        self.position.y = trans.transform.translation.y
        self.orientation.z = trans.transform.rotation.z
        left_steps = MAX_STEPS - self.steps_in_episode
        numeric_state = np.array(
            [
                self.position.x,
                self.position.y,
                self.orientation.z,
                self.last_action,
                left_steps,
                self.map_completeness_pct,
            ]
        )

        return numeric_state

    def _infer_reward(self) -> None:
        """ """

        if self.map_completeness_pct > MAP_COMPLETENESS_THRESHOLD:
            self.reward = REWARD_MAP_COMPLETED
            state = "comp"
        elif self.min_distance < COLLISION_THRESHOLD:
            # Robot likely hit the wall
            self.reward = REWARD_CRASHED
            state = "crashed"
        else:
            self.reward = self.map_completeness_pct - self.last_map_completeness_pct
            state = ""

        self.reward_in_episode += self.reward
        # rospy.loginfo("reward:" + str(self.reward) + "state: " + state)

    def _update_map_completeness(self) -> None:

        # rospy.loginfo("waiting map")
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/map", OccupancyGrid, timeout=TIMEOUT)
            except:
                pass
        # rospy.loginfo("ended waiting map")

        self.occupancy_grid = data.data
        sum_grid = len(self.occupancy_grid)
        num_occupied = 0
        num_unoccupied = 0
        num_negative = 0
        for n in self.occupancy_grid:
            if n == 0:
                num_unoccupied += 1
            elif n == 100:
                num_occupied += 1

        self.last_map_completeness_pct = self.map_completeness_pct
        self.map_completeness_pct = (
            (num_occupied + num_unoccupied) * 100 / sum_grid
        ) / self.map_size_ratio
        if self.steps_in_episode == 1:
            self.last_map_completeness_pct = self.map_completeness_pct

    def _send_action(self, steering: float, throttle: float) -> None:
        speed = Twist()
        speed.angular.z = steering
        speed.linear.x = throttle
        self.ack_publisher.publish(speed)

    def render(self, mode="human") -> None:
        """
        unused function
        """
        raise NotImplementedError
        return

    def callback_odom(self):
        """update odom(word pose) and pose_map(map pose)
        Args:
            Odometry (_type_): _description_
        """
        _Odometry = None
        while _Odometry is None:
            try:
                _Odometry = rospy.wait_for_message(
                    "/steer_drive_controller/odom", Odometry
                )
            except:
                rospy.sleep(0.2)
        position_x = _Odometry.pose.pose.position.x
        position_y = _Odometry.pose.pose.position.y
        position_z = _Odometry.pose.pose.position.z
        orientation_z = _Odometry.pose.pose.orientation.z
        orientation_w = _Odometry.pose.pose.orientation.w
        # heading = math.atan2(2*orientation_w*orientation_z,1-2*orientation_z**2)
        ##print'odom x: %s' % position_x
        ##print'odom y: %s' % position_y
        self.odom = np.array([position_x, position_y, orientation_z, orientation_w])

    def callback_odom2(self, Odometry):
        """update odom(word pose) and pose_map(map pose)
        Args:
            Odometry (_type_): _description_
        """
        _Odometry = Odometry

        position_x = _Odometry.pose.pose.position.x
        position_y = _Odometry.pose.pose.position.y
        position_z = _Odometry.pose.pose.position.z
        orientation_z = _Odometry.pose.pose.orientation.z
        orientation_w = _Odometry.pose.pose.orientation.w
        self.odom = np.array([position_x, position_y, orientation_z, orientation_w])

    def getcurrentpose(self):
        """get `pose_map`(pose in map frame) using tf /map and /odom"""
        tf_trans = None
        while tf_trans is None:
            try:
                # listen to transform
                tf_trans, tf_rot = self.tflistener.lookupTransform(
                    "/map", "/odom", rospy.Time(0)
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.sleep(0.2)
        theta_tf = math.atan2(2 * tf_rot[3] * tf_rot[2], 1 - 2 * tf_rot[2] ** 2)
        theta_odom = math.atan2(
            2 * self.odom[3] * self.odom[2], 1 - 2 * self.odom[2] ** 2
        )  # rad
        x_tf = tf_trans[0]
        y_tf = tf_trans[1]
        T_tf = np.array(
            [
                [math.cos(theta_tf), -math.sin(theta_tf), x_tf],
                [math.sin(theta_tf), math.cos(theta_tf), y_tf],
                [0, 0, 1],
            ]
        )
        pose_odom = np.array([self.odom[0], self.odom[1], 1])
        positioninmap = T_tf.dot(pose_odom)
        self.pose_map = positioninmap
        self.pose_map[2] = theta_odom + theta_tf  # nparray[x,y,theta(rad)]
        self.pose_map[2] = (
            self.pose_map[2] * 180 / math.pi
        )  # nparray[x,y,theta(degree)]
        ##printself.pose_map
        return self.pose_map

    def process(self):
        """
        get next_goal_world in serval metrix:closest goal/max info-gain/max-utility \n
        cal three type of cells;\n
        detect centor of frontier and group by distance to courent pose;\n
        cal metrix using trajectory;\n
        Args:
            info (self.p0):
            grid (self.rawmap):
            odom (self.pose_map):
        Returns:
            _type_:next_goal_world/m
        """
        map_msg = self.occmap

        resolution = map_msg.info.resolution
        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y
        data = map_msg.data
        grid = np.array(data).reshape(map_msg.info.height, map_msg.info.width)
        size = grid.shape
        self.callback_odom()
        robot_pose_world = self.getcurrentpose()  # robot initial position in world
        robot_pose_pixel = [0, 0]  # robot initial position in grid (pixel in image)
        robot_pose_pixel[0] = (robot_pose_world[0] - origin_x) / resolution
        robot_pose_pixel[1] = (robot_pose_world[1] - origin_y) / resolution
        # --------------------------------------------- unvisited cells ---------------------
        thresh_low = -1
        thresh_high = -1
        unknown = ((grid <= thresh_high) & (grid >= thresh_low)) * 1.0  # threshold
        self.unkonow_area = np.sum(unknown)
        # ------------------------------frontier detection----------------------------------
        # frontier_cells = [x for x in contours_unvisited_cell if x in contours_open_cell]
        frontier_cells = []
        for i in range(1, map_msg.info.height - 1):
            for j in range(1, map_msg.info.width - 1):
                if grid[i, j] == 0:
                    near = grid[i - 1 : i + 2, j - 1 : j + 2].reshape(1, -1)[0]
                    cnt = 0
                    for near_i in list(near):
                        cnt = cnt + 1 if near_i == -1 else cnt
                    if cnt >= 3 and i >= 3 and j >= 3:
                        grid_near = grid[i - 3 : i + 4, j - 3 : j + 4].reshape(1, -1)[0]
                        if not grid_near.__contains__(100):
                            frontier_cells.append([i, j])
        grid_frontier = np.zeros(size)
        for ele in frontier_cells:
            grid_frontier[int(ele[0]), int(ele[1])] = 1
        # group them!
        conected_frontier, label_num = measure.label(
            grid_frontier, background=0, return_num=True
        )
        manh_dist = []  # stores distances
        cents = []  # stores centers of frontiers
        for region in regionprops(conected_frontier):
            # take regions with large enough areas
            # FIXME:change region threshod
            if region.area >= 10:  # 30 do not consider small frontier groups
                # the centroid of each valid region
                cen_y = region.centroid[0]  # Centroid coordinate tuple (row, col)
                cen_x = region.centroid[1]  # Centroid coordinate tuple (row, col)
                cents.append([cen_x, cen_y])  # cents[col,row]
                # Manhattan Distance from robot to each frontier center
                manh = abs(cen_x - robot_pose_pixel[0]) + abs(
                    cen_y - robot_pose_pixel[1]
                )
                manh_dist.append(manh)

        # sort two list: centers of each frontiers according to the man_distance
        # a sorted record of all the candidate goals (close-far)
        cents_sorted = [x for _, x in sorted(zip(manh_dist, cents))]
        cents_sorted_world = 1 * cents_sorted
        for ele in cents_sorted_world:
            ele[0] = ele[0] * resolution + origin_x
            ele[1] = ele[1] * resolution + origin_y
        frontier_cells_world = []
        for ele in frontier_cells:
            frontier_cells_world.append(
                [ele[1] * resolution + origin_y, ele[0] * resolution + origin_x]
            )

        # sort front cells by distance
        dist_fronts = []
        for front in frontier_cells_world:
            dist_fronts.append(self.distance(self.pose_map[0:2], front))
        frontier_cells_word_sorted = [
            f for _, f in sorted(zip(dist_fronts, frontier_cells_world))
        ]

        self.cents_sorted_world = cents_sorted_world
        random.shuffle(frontier_cells_world)
        self.frontier_cells = frontier_cells_world
        self.frontier_cells_sorted = frontier_cells_word_sorted

    def setupGoals(self, next_x, next_y):
        """set self.goals[1] based on next_x, next_y; \n
        set self.flag_stuck = 1 accorading to  goals[1] before

        Args:
            next_x (_type_): _description_
            next_y (_type_): _description_
        """

        # FIXME: change the tables below
        goalB = Pose()  # goalB is next frontier
        goalB.position.x = next_x
        goalB.position.y = next_y
        # goalB.orientation.z = self.odom[2]
        goalB.orientation.z = random.uniform(0, 1)
        # goalB.orientation.w = self.odom[3]
        goalB.orientation.w = math.sqrt(1 - goalB.orientation.z**2)
        return goalB

    def navigateToGoal(self, goal_pose: Pose):
        """using action move_base to move to goal pose;\n
        wait until achieve by using  `client.get_state()`

        Args:
            goal_pose (Pose): _description_

        Returns:
            _type_: _description_
        """
        # Create the goal point
        goal = move_base_msgs.MoveBaseGoal()
        goal.target_pose.pose = goal_pose
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = "map"  # odom,  map
        # Send the goal!
        # print"sending goal"
        self.client.send_goal(goal)
        # print"waiting for result"
        """PENDING = 0, ACTIVE = 1, PREEMPTED = 2, SUCCEEDED = 3"""
        rospy.loginfo("keep_waiting for finish the goal.")

    def resetCostmaps(self):
        if self.clear_costmap_srv is None:
            rospy.wait_for_service("/move_base/clear_costmaps")
            self.clear_costmap_srv = rospy.ServiceProxy(
                "/move_base/clear_costmaps", std_srvs.Empty
            )
        self.clear_costmap_srv()

    def valid_act(self, act):
        for cen in self.cents_sorted_world:
            gred = (
                math.atan2(cen[1] - self.odom[1], cen[0] - self.odom[0]) * 180 / math.pi
            )
            gred = gred + 360 if gred < 0 else gred
            gred = gred - 360 if gred > 360 else gred
            if act == gred // 90:
                return cen
        return None

    def valid_act_cell(self, act):
        for cell in self.frontier_cells:
            gred = (
                math.atan2(cell[1] - self.odom[1], cell[0] - self.odom[0])
                * 180
                / math.pi
            )
            gred = gred + 360 if gred < 0 else gred
            gred = gred - 360 if gred > 360 else gred
            if act == gred // 90:
                return cell
        return None

    def valid_act_list(self):
        act_index = []
        for cen in self.frontier_cells:
            gred = (
                math.atan2(cen[1] - self.odom[1], cen[0] - self.odom[0]) * 180 / math.pi
            )
            gred = gred + 360 if gred < 0 else gred
            gred = gred - 360 if gred > 360 else gred
            act_index.append(gred // 90)
        return act_index

    def set_plan_goal(self, center):
        """get start pose and goal pose(center)
        Args:
            center (_type_): _description_
        Returns:
            start, goal, tolerance: _description_
        """
        # set start and goal
        self.process()
        start = PoseStamped()
        start.header.frame_id = "map"
        start.pose.position.x = self.pose_map[0]
        start.pose.position.y = self.pose_map[1]
        start.pose.orientation.z = self.odom[2]
        start.pose.orientation.w = self.odom[3]
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.position.x = center[0]
        goal.pose.position.y = center[1]
        goal.pose.orientation.z = self.odom[2]
        goal.pose.orientation.w = self.odom[3]
        # set tolerance
        tolerance = 0.1
        return start, goal, tolerance

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:

        self.last_action = self.now_action
        self.now_action = action

        # initialize rewards, next_state, done
        self.reward = None
        self.done = False
        self.steps_in_episode += 1
        self.total_step += 1

        self.process()
        last_unkonw_area = self.unkonow_area
        cen = self.valid_act_cell(action)
        if cen is not None:
            rospy.logwarn("find frontiers in this action")
            rospy.logwarn("go to: {}".format(cen))
            goal = self.setupGoals(cen[0], cen[1])
            self.navigateToGoal(goal)

        else:
            cnt = 0
            while cen is None:
                try:
                    rospy.logwarn("no frontiers in this action")
                    cen = (
                        self.cents_sorted_world[0]
                        if len(self.cents_sorted_world) > 0
                        else self.frontier_cells_sorted[
                            len(self.frontier_cells_sorted) // 2
                        ]
                    )
                except IndexError:
                    self.process()
                    rospy.sleep(1)
                    cnt += 1
                    if cnt > 5:
                        if len(self.frontier_cells) < 8:
                            rospy.logwarn("len(self.frontier_cells)<8")
                            self.done = True
                            self.reward = 2
                            dt_now = datetime.datetime.now()
                            fpath = "/home/nianfei/ros_ws/src/dqn_for_slam/dqn_for_slam/log/maze5_trace/test_trace_{}_{}{}{}.txt".format(
                                self.total_step, dt_now.month, dt_now.day, dt_now.hour
                            )
                            with open(fpath, "w+") as f:
                                f.write(pprint.pformat(self.full_trace))
                            return self.get_state(), self.done, self.reward, {}
                        else:
                            self.done = True
                            self.reward = -1
                            rospy.logerr("find no fronts,set reward -1")
                            rospy.logerr(
                                "step:{},\nreward :{},\n total reward:{}".format(
                                    self.total_step, self.reward, self.reward_in_episode
                                )
                            )
                            return self.get_state(), self.reward, self.done, {}
            rospy.logwarn("go to nearest central front: {}".format(cen))
            goal = self.setupGoals(cen[0], cen[1])
            self.navigateToGoal(goal)

        self.odom_dist = []
        self.trace_instep = []

        while str(self.client.get_state()) != "3":
            self.process()
            map_increase = last_unkonw_area - self.unkonow_area

            # stop if cen has no near frontiers
            cen_near = 0
            for p in self.frontier_cells:
                cen_near = cen_near + 1 if self.distance(cen, p) < 1 else cen_near
            if cen_near < 1:
                rospy.logwarn("cen has no near frontiers")
                break
            if map_increase > 1000:
                rospy.logwarn("map_increase > 1000")
                break
            if len(self.frontier_cells) < 8:
                rospy.logwarn("len(self.frontier_cells)<8")
                self.done = True
                dt_now = datetime.datetime.now()
                fpath = "/home/nianfei/ros_ws/src/dqn_for_slam/dqn_for_slam/log/maze5_trace/test_trace_{}_{}{}{}.txt".format(
                    self.total_step, dt_now.month, dt_now.day, dt_now.hour
                )
                with open(fpath, "w+") as f:
                    f.write(pprint.pformat(self.full_trace))
                break

            self.trace_instep.append(self.pose_map.tolist())
            l = len(self.trace_instep) - 1
            if l:
                self.odom_dist.append(
                    self.distance(
                        self.trace_instep[l - 1][0:2], self.trace_instep[l][0:2]
                    )
                )
            # 5l -0.2/0.5m 0.04/0.1m retate:0.01
            if l % 5 == 0:
                rospy.logwarn(
                    "plan length:{};average distance:{:.3f};front:{}".format(
                        l,
                        sum(self.odom_dist[-10:]) / 10,
                        len(self.frontier_cells_sorted),
                    )
                )
            if l > 200 and sum(self.odom_dist[-3:]) / 3 < 0.05:
                self.stuck_pose.append(self.pose_map[0:2])
                if len(self.stuck_pose) >= 6:
                    rospy.logwarn("stuck 3 times , reset")
                    self.stuck_pose = []
                    self.done = True
                    break
                rospy.loginfo(
                    "stuck from {} to {} ".format(
                        self.pose_map[0:2],
                        cen,
                    )
                )
                break
            if l > 150 and sum(self.odom_dist[-30:]) / 30 < 0.01:
                rospy.loginfo(
                    "early stuck from {} to {} ".format(
                        self.pose_map[0:2],
                        cen,
                    )
                )
                break
            rospy.sleep(0.2)

        self.client.cancel_all_goals()
        self.resetCostmaps()
        # wait for gmapping
        rospy.logwarn("this action finished without interupt")
        rospy.sleep(1)
        next_state = self.get_state()
        self.reward = -sum(self.odom_dist) * 0.1
        self.reward = -1 if self.reward > -0.05 else self.reward
        self.reward = 2 if len(self.frontier_cells) < 10 else self.reward
        self.reward_in_episode += self.reward
        self.full_trace.append(self.trace_instep)
        info = {}
        rospy.logerr(
            "step:{},\nreward :{},\n total reward:{}".format(
                self.total_step, self.reward, self.reward_in_episode
            )
        )
        return next_state, self.reward, self.done, info

    def get_state(self):
        map_msg = self.occmap

        resolution = map_msg.info.resolution
        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y
        data = map_msg.data
        # grid = np.array(data).reshape(map_msg.info.height, map_msg.info.width)
        grid = np.array(data)
        # size = grid.shape
        self.callback_odom()
        self.getcurrentpose()  # robot initial position in world
        # sampled_grid = grid[0 : size[0] : size[0] // 100, 0 : size[1] : size[1] // 100]

        size = grid.size
        sampled_grid = grid[0 : size - 1 : size // 8100][3:8100]
        # next_state = sampled_grid
        next_state = np.concatenate((sampled_grid, np.array(self.pose_map)))
        return next_state

    def nf_nav(self):
        self.reset()
        self.full_trace = []
        self.stuck_pose = [[0, 0]]
        index = 0
        flag_s = 0
        while len(self.frontier_cells_sorted) > 5:
            self.process()
            last_unkonw_area = self.unkonow_area
            try:
                """method 1"""
                cen = (
                    self.cents_sorted_world[0]
                    if len(self.cents_sorted_world) > 0
                    else self.frontier_cells_sorted[
                        len(self.frontier_cells_sorted) // 3
                    ]
                )
                """method 2"""
                # if len(self.cents_sorted_world) > 0:
                #     for f in self.cents_sorted_world:
                #         for d in self.stuck_pose:
                #             if self.distance(d, f) > 0.5:
                #                 cen = f
                # if cen is None:
                #     for f in self.frontier_cells_sorted:
                #         for d in self.stuck_pose:
                #             if self.distance(d, f) > 0.5:
                #                 cen = f
            except IndexError:
                self.process()
                rospy.sleep(1)
                cnt += 1
                if cnt > 5:
                    self.reset()

            rospy.logwarn("go to nearest central front: {}".format(cen))
            goal = self.setupGoals(cen[0], cen[1])
            self.navigateToGoal(goal)
            self.odom_dist = []
            self.trace_instep = []

            while str(self.client.get_state()) != "3":
                self.process()
                map_increase = last_unkonw_area - self.unkonow_area

                # stop if cen has no near frontiers
                cen_near = 0
                for p in self.frontier_cells:
                    cen_near = cen_near + 1 if self.distance(cen, p) < 1 else cen_near
                if cen_near < 1:
                    rospy.logwarn("cen has no near frontiers")
                    break

                if map_increase > 2000:
                    rospy.logwarn("map_increase > 2000")
                    break
                if len(self.frontier_cells) == 0:
                    rospy.logwarn("len(self.frontier_cells)<10")
                    break

                self.trace_instep.append(self.pose_map.tolist())
                l = len(self.trace_instep) - 1
                if l:
                    self.odom_dist.append(
                        self.distance(
                            self.trace_instep[l - 1][0:2], self.trace_instep[l][0:2]
                        )
                    )
                if l % 5 == 0:
                    rospy.logwarn("{} to stop".format(100 - l))
                if l > 100 and sum(self.odom_dist[-20:]) / 20 < 0.05:
                    flag_s = 1
                    rospy.logwarn("break 100")
                    break
                rospy.sleep(0.05)
            l = len(self.trace_instep) - 1
            index = index + 1 if flag_s else 0
            flag_s = 0
            self.full_trace.append(self.trace_instep)
            self.client.cancel_all_goals()
            self.resetCostmaps()
            # wait for gmapping
            rospy.logwarn("stop this action")

        with open("src/dqn_for_slam/dqn_for_slam/log/trace_maze5.npy", "a+") as f:
            f.write(pprint.pformat(self.full_trace))

    def nf_nav2(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.last_action = self.now_action
        self.now_action = action

        # initialize rewards, next_state, done
        self.reward = None
        self.done = False
        self.steps_in_episode += 1
        self.total_step += 1
        self.process()
        last_unkonw_area = self.unkonow_area
        cen = None
        if len(self.frontier_cells) < 3:
            rospy.logwarn("len(self.frontier_cells)<5")
            self.done = True
            dt_now = datetime.datetime.now()
            fpath = "/home/nianfei/ros_ws/src/dqn_for_slam/dqn_for_slam/log/maze5_trace/nf_trace_{}_{}{}{}.txt".format(
                self.total_step, dt_now.month, dt_now.day, dt_now.hour
            )
            with open(fpath, "w+") as f:
                f.write(pprint.pformat(self.full_trace))
            return 0
        if cen is not None:
            rospy.logwarn("find frontiers in this action")
            rospy.logwarn("go to: {}".format(cen))
            goal = self.setupGoals(cen[0], cen[1])
            self.navigateToGoal(goal)

        else:
            cnt = 0
            while cen is None:
                try:
                    rospy.logwarn("no frontiers in this action")
                    cen = (
                        self.cents_sorted_world[0]
                        if len(self.cents_sorted_world) > 0
                        else self.frontier_cells_sorted[
                            len(self.frontier_cells_sorted) // 5
                        ]
                    )
                except IndexError:
                    self.process()
                    rospy.sleep(1)
                    cnt += 1
            rospy.logwarn("go to nearest central front: {}".format(cen))
            goal = self.setupGoals(cen[0], cen[1])
            self.navigateToGoal(goal)

        self.odom_dist = []
        self.trace_instep = []

        while str(self.client.get_state()) != "3":
            self.process()
            map_increase = last_unkonw_area - self.unkonow_area

            # stop if cen has no near frontiers
            cen_near = 0
            for p in self.frontier_cells:
                cen_near = cen_near + 1 if self.distance(cen, p) < 1 else cen_near
            if cen_near < 1:
                rospy.logwarn("cen has no near frontiers")
                break
            if map_increase > 1000:
                rospy.logwarn("map_increase > 1000")
                break
            if len(self.frontier_cells) < 3:
                rospy.logwarn("len(self.frontier_cells)<5")
                self.done = True
                dt_now = datetime.datetime.now()
                fpath = "/home/nianfei/ros_ws/src/dqn_for_slam/dqn_for_slam/log/mazetest_trace/nf_trace_{}_{}{}{}.txt".format(
                    self.total_step, dt_now.month, dt_now.day, dt_now.hour
                )
                with open(fpath, "w+") as f:
                    f.write(pprint.pformat(self.full_trace))
                break

            self.trace_instep.append(self.pose_map.tolist())
            l = len(self.trace_instep) - 1
            if l:
                self.odom_dist.append(
                    self.distance(
                        self.trace_instep[l - 1][0:2], self.trace_instep[l][0:2]
                    )
                )
            # 5l -0.2/0.5m 0.04/0.1m retate:0.01
            if l % 5 == 0:
                rospy.logwarn(
                    "plan length:{};average distance:{:.3f};front:{}".format(
                        l,
                        sum(self.odom_dist[-10:]) / 10,
                        len(self.frontier_cells_sorted),
                    )
                )
            if l > 100 and sum(self.odom_dist[-10:]) / 10 < 0.05:
                self.stuck_pose.append(self.pose_map[0:2])
                rospy.loginfo(
                    "stuck from {} to {} ".format(
                        self.pose_map[0:2],
                        cen,
                    )
                )
                break
            rospy.sleep(0.2)

        self.client.cancel_all_goals()
        self.resetCostmaps()
        # wait for gmapping
        rospy.logwarn("this action finished without interupt")
        rospy.sleep(1)
        next_state = self.get_state()
        self.reward = -sum(self.odom_dist) * 0.1
        self.reward = -1 if self.reward > -0.05 else self.reward
        self.reward = 2 if len(self.frontier_cells) < 10 else self.reward
        self.reward_in_episode += self.reward
        self.full_trace.append(self.trace_instep)
        info = {}
        rospy.logerr(
            "step:{},\nreward :{},\n total reward:{}".format(
                self.total_step, self.reward, self.reward_in_episode
            )
        )
        return next_state, self.reward, self.done, info
