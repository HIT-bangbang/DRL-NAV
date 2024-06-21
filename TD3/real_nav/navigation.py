import math

import time
from os import path
from numpy import inf

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from squaternion import Quaternion

TIME_DELTA = 0.01
GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.1

class Navitation:
    def __init__(self,environment_dim):

        self.environment_dim = environment_dim
        self.velodyne_data = np.ones(self.environment_dim) * 10     #存储激光雷达数据，填充为10（因为最大取10米内的数据为有效数据）
        self.last_odom = None                #上一时刻的激光雷达数据
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.odom_x = 0                             #机器人开始时刻的位置
        self.odom_y = 0
        self.done = False
        
        
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)      #发布速度
        self.laser_sub = rospy.Subscriber(                                                # 订阅激光雷达的点云数据
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom_sub = rospy.Subscriber(                                                      #订阅机器人里程计
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )
        # self.cmd_sub = rospy.Subscriber(                                                      #订阅机器人里程计
        #     "/cmd_vel", Twist, self.cmd_callback, queue_size=1
        # )
        self.goal_sub = rospy.Subscriber(                                                      #订阅机器人里程计
            "/move_base_simple/goal",PoseStamped, self.goal_callback, queue_size=1
        )

        # 记录每个区间的起始角度和终止角度，这个在两边加上的0.03是为什么呢？
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03




        # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
        # range as state representation
        # 激光雷达回调函数
    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break


    # 里程计回调函数
    def odom_callback(self, od_data):
        self.last_odom = od_data    #记录过去的里程计数据
        # print(self.last_odom)
    
    def goal_callback(self, goal):
        self.goal_x = goal.pose.position.x
        self.goal_y = goal.pose.position.y
        self.done = False

    def getstate(self):
        return self.done

    def step(self,action):

        target = False
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)                              # 暂停程序，让机器人执行完毕当前动作
        # self.stop()             #执行完毕就停
        
        # read velodyne laser state
        self.done, collision, min_laser = self.observe_collision(self.velodyne_data) #是否碰撞
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]     #这个是要喂到经验池里的
        

        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)  # 用弧度表示的欧拉角
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            self.done = True

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(target, collision, action, min_laser)
        return state, reward, self.done, target

    def stop(self):
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        self.vel_pub.publish(stop_cmd)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            return 100.0
        elif collision:
            return -5.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
