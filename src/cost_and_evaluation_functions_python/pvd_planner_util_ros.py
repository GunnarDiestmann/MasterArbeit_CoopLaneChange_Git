import rospy
import tf
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Point


def quaternion_from_yaw(yaw):
    orientation = Quaternion()
    quat = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
    orientation.x = quat[0]
    orientation.y = quat[1]
    orientation.z = quat[2]
    orientation.w = quat[3]
    return orientation


def position_from_x_y(x, y):
    position = Point()
    position.x = x
    position.y = y
    position.z = 0.0
    return position
