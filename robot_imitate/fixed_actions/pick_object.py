import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int32

import tf2_ros
from enum import Enum
import time
import time


class OperationState(Enum):
    IDLE = 0
    GO_CLOSE = 1
    CLOSE_GRIPPER = 2
    PICK_UP = 3
    OPEN_GRIPPER = 4
    END = 5



class PickObject(Node):

    def __init__(self, sim):
        super().__init__('cmd_vel_publisher')
        self.get_logger().info("PickObject node started")


        self.timer = self.create_timer(0.1, self.publish_cmd_vel)

        self.publisher_speed_limiter = self.create_publisher(
            PoseStamped, '/target_frame_raw', 1)

        self.publisher_gripper = self.create_publisher(
            Float64MultiArray, '/position_controller/commands', 1)

        self.current_pose_subscriber = self.create_subscription(
            PoseStamped,
            '/current_pose',
            self.current_pose_callback,
            1)
        self.current_pose_subscriber

        self.end_signal_subscriber = self.create_subscription(
            Int32,
            '/activate_fix_sequence',
            self.end_signal_callback,
            1)
        self.end_signal_subscriber

        self.frames = []
        self.twist_msg = Twist()

        self.counter = 0
        self.sim = sim
        self.gripper_state = 0

        self.observation_pose = None
        self.current_pose = None
        self.gripper_msg = Float64MultiArray()

        self.timer = time.time()

        self.state = OperationState.IDLE

        self.activate_sequence = False

        if not sim:
            self.switcher_publisher = self.create_publisher(Int32, '/gripper_switcher', 1)
        
    
    def get_transform(self, target_frame, source_frame):
        try:
            transform = self.tfBuffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time())
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Failed to get transform: {e}")
            return None


    def current_pose_callback(self, msg):
        self.current_pose = msg

    def  end_signal_callback(self, msg):
        if msg.data == 1:
            self.activate_sequence = True
        
          
    def publish_cmd_vel(self):
        if not self.activate_sequence:
            return
    
        if self.current_pose is None:
            return
    
        self.get_logger().info(f'========================={self.state}=========================')
        self.current_pose.header.stamp = self.get_clock().now().to_msg()
        self.current_pose.header.frame_id = 'link_base'

    
        if self.state == OperationState.IDLE:
            self.state = OperationState.OPEN_GRIPPER
            self.timer = time.time()

        elif self.state == OperationState.OPEN_GRIPPER:
            # open later because make noise
            if not self.sim:
                msg = Int32()
                msg.data = 1
                self.switcher_publisher.publish(msg)

            self.state = OperationState.GO_CLOSE
            self.timer = time.time()

        elif self.state == OperationState.GO_CLOSE:
                self.current_pose.pose.position.z = 0.085 # 0.085
                self.publisher_speed_limiter.publish(self.current_pose)
                if time.time() - self.timer > 2.5:
                    self.state = OperationState.CLOSE_GRIPPER
                    self.timer  =time.time()
        
        elif self.state == OperationState.CLOSE_GRIPPER:
            if self.sim:
                self.gripper_msg.data = [-0.01]
                self.publisher_gripper.publish(self.gripper_msg)
            else:
                msg = Int32()
                msg.data = 0
                self.switcher_publisher.publish(msg)

            if time.time() -  self.timer > 1.0:
                self.state = OperationState.PICK_UP
                self.timer = time.time()
        elif self.state == OperationState.PICK_UP:
                self.current_pose.pose.position.z = 0.29
                self.publisher_speed_limiter.publish(self.current_pose)
                if time.time() - self.timer > 1.5:
                    self.state = OperationState.END
                    self.timer = time.time()

        elif self.state == OperationState.END:

            if time.time() - self.timer > 2.5:
                self.activate_sequence = False
                self.state = OperationState.IDLE
                if not self.sim:
                    # shutdown gripper
                    msg = Int32()
                    msg.data = 2
                    self.switcher_publisher.publish(msg)
            

def main(args=None):

    import argparse
    parser = argparse.ArgumentParser(description='PickObject')
    parser.add_argument('--sim', action='store_true', help='Use simulation')
    parsed_args = parser.parse_args()

    rclpy.init(args=args)
    cmd_vel_publisher = PickObject(parsed_args.sim)
    rclpy.spin(cmd_vel_publisher)
    cmd_vel_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
