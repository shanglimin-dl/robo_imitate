import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped, Twist, Pose
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Int32

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import torch
from common.diffusion_policy import DiffusionPolicy
import tf2_ros

from enum import Enum
import subprocess
import re
import time
import subprocess
import re
from collections import deque
import transforms3d as t3d
import time
from pathlib import Path
import torch
import cv2

from ros2_numpy_tf import numpy2ros, ros2numpy


def call_ros2_service(activate_controllers, deactivate_controllers):
    service_name = '/controller_manager/switch_controller'
    service_type = 'controller_manager_msgs/srv/SwitchController'
    strictness = '2'
    activate_asap = 'true'

    command = f'ros2 service call {service_name} {service_type} "{{activate_controllers: [\"{activate_controllers}\"], deactivate_controllers: [\"{deactivate_controllers}\"], strictness: {strictness}, activate_asap: {activate_asap}}}"'
    try:
        result = subprocess.run(command, shell=True,
                                check=True, capture_output=True, text=True)
        match = re.search(r'response:\n(.*)', result.stdout, re.DOTALL)
        print(f"{activate_controllers}:", match.group(1).strip())
    except subprocess.CalledProcessError as e:
        print(f"Error calling ROS 2 service: {e}")


from http.server import BaseHTTPRequestHandler, HTTPServer
import os

class MJPEGStreamHandler(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server, frame_queue):
        self.frame_queue = frame_queue
        self.streaming_started = False  # Flag to track streaming start

        if not os.path.exists('images'):
            os.makedirs('images')
        super().__init__(request, client_address, server)

    def do_GET(self):
        if self.path == '/':
            if not self.streaming_started:
                self.start_streaming()
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            while True:
                try:
                    frame = self.frame_queue.get()

                    _, img_encoded = cv2.imencode('.jpg', frame)
                    frame_bytes = img_encoded.tobytes()

                    # Send the frame to the browser
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', len(frame_bytes))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n--frame\r\n')
                except Exception as e:
                    print("Exception: ", e)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

    def start_streaming(self):
        self.streaming_started = True

    def start_streaming(self):
        self.streaming_started = True


def start_stream_server(frame_queue):
    ip_address = '0.0.0.0'
    server = HTTPServer((ip_address, 8080), lambda *args, **kwargs: MJPEGStreamHandler(*args, **kwargs, frame_queue=frame_queue))
    print(f'Starting streaming server on http://'+ ip_address +':8080/')
    server.serve_forever()



class OperationState(Enum):
    IDLE = 0
    INFERENCE = 1
    GO_CLOSE = 2
    CLOSE_GRIPPER = 3
    PICK_UP = 4
    OPEN_GRIPPER = 5
    END = 6


def check_pose_stamped_values(pose_stamped_msg):
    position = pose_stamped_msg.pose.position
    orientation = pose_stamped_msg.pose.orientation
    is_position_zero = position.x == 0.0 and position.y == 0.0 and position.z == 0.0
    
    is_orientation_zero_except_w = orientation.x == 0.0 and orientation.y == 0.0 and orientation.z == 0.0 and orientation.w == 1.0
    
    return is_orientation_zero_except_w


def is_end_of_episode(action):
    arr = np.array([action[0], action[1], action[2], action[3], action[4], action[5], action[6]])
    close_to_zero = np.isclose(arr, 0, atol=0.05)
    
    return np.all(close_to_zero)


def plot_action_trajectory(sim, positions1, positions2=None):
    import matplotlib.pyplot as plt
    from datetime import datetime
    import os

    positions1 = np.array(positions1)
    norm1 = plt.Normalize(positions1[:, 2].min(), positions1[:, 2].max())
    print(norm1)
    colors1 = plt.cm.viridis(norm1(positions1[:, 2]))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc1 = ax.scatter(positions1[:, 0], positions1[:, 1], positions1[:, 2], c=colors1, marker='o', s=50, alpha=0.8, edgecolor='k', linewidth=0.5, label='Trajektorija observacije')

    if positions2 is not None:
        positions2 = np.array(positions2)
        norm2 = plt.Normalize(positions2[:, 2].min(), positions2[:, 2].max())
        colors2 = plt.cm.plasma(norm2(positions2[:, 2]))
        print(norm2)

        sc2 = ax.scatter(positions2[:, 0], positions2[:, 1], positions2[:, 2], c=colors2, marker='^', s=50, alpha=0.8, edgecolor='k', linewidth=0.5, label='Trajektorija akcije')

    ax.set_xlabel('X osa', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y osa', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z osa', fontsize=12, fontweight='bold')

    ax.set_title('Razlika između akcije i observacije', fontsize=14, fontweight='bold')

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # ax.view_init(elev=20, azim=-30)
    ax.legend(loc='best')

    now = datetime.now()
    real_path = 'action_trajectoris/real/'
    sim_path = 'action_trajectoris/sim/'

    if not os.path.exists('action_trajectoris'):
        os.mkdir('action_trajectoris')
    if not os.path.exists(real_path):
        os.mkdir(real_path)
    if not os.path.exists(sim_path):
        os.mkdir(sim_path)

    name = real_path + now.strftime("%Y_%m_%d_%H_%M_%S") + '.png'
    if sim:
        name = sim_path + now.strftime("%Y_%m_%d_%H_%M_%S") + '.png'

    plt.savefig(name, bbox_inches='tight', dpi=300)

    # plt.show()


class CmdVelPublisher(Node):

    def __init__(self, sim):
        super().__init__('cmd_vel_publisher')
        self.get_logger().info("CmdVelPublisher node started")

        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.listener_callback,
            1)
        self.bridge = CvBridge()

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
        

        self.x_translation = 0.0
        self.y_translation = 0.0
        self.z_translation = 0.0
        

        self.output_directory = Path("imitation/outputs/example")
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)

        self.device = torch.device("cuda")
        self.pretrained_policy_path = Path("imitation/outputs/train")

        self.policy = DiffusionPolicy.from_pretrained(self.pretrained_policy_path)
        print(self.policy)
        self.policy.eval()
        self.policy.to(self.device)
        
        self.policy.reset()
        self.current_pose_relativ = PoseStamped()
        self.current_pose = PoseStamped()
        self.step = 0
        
        self.image  = None
        self.action = None

        self.current_x =  0.0
        self.current_y =  0.0
        self.current_z =  0.0
        # 60 steps for sim 90 for real env
        self.max_episode_steps = 90
        self.frames = []
        self.twist_msg = Twist()

        self.counter = 0
        self.queue = deque(maxlen=3)
        self.sim = sim
        self.gripper_state = 0

        self.observation_pose = None
        self.observation_current_pose = PoseStamped()
        if self.sim:
            # on init open griper and move screwdriver on random position
            self.gripper_msg = Float64MultiArray()
            self.gripper_msg.data = [0.0]
            self.publisher_gripper.publish(self.gripper_msg)

            self.publisher_respawn = self.create_publisher(Twist, '/respawn', 1)
            msg_arr = Twist()
            self.publisher_respawn.publish(msg_arr)

        self.timer = time.time()

        self.publisher_joint_init = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 1)
        self.joint_state = JointTrajectory()
        self.joint_names = ['joint1', 'joint2',
                            'joint3', 'joint4', 'joint5', 'joint6']

        point = JointTrajectoryPoint()
        point.positions = [0.00148, 0.06095, 1.164, -0.00033, 1.122, -0.00093]
        point.time_from_start.sec = 3
        point.time_from_start.nanosec = 0

        self.joint_state.points = [point]
        self.joint_state.joint_names = self.joint_names

        if not self.sim:
            self.switcher_publisher = self.create_publisher(Int32, '/gripper_switcher', 1)
            msg = Int32()
            msg.data = 1
            self.switcher_publisher.publish(msg)

            time.sleep(1)
            msg = Int32()
            msg.data = 2
            self.switcher_publisher.publish(msg)
        # move arm to init pose
        call_ros2_service('joint_trajectory_controller',
                              'cartesian_motion_controller')
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.publisher_joint_init.publish(self.joint_state)

        time.sleep(3)
        call_ros2_service('cartesian_motion_controller', 'joint_trajectory_controller')
        
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer, self)

        self.state = OperationState.IDLE
        self.observation_msg = PoseStamped()
        self.start_pose = None


        self.plotting_observations = []
        self.plotting_actions = []
        
    
    def get_transform(self, target_frame, source_frame):
        try:
            transform = self.tfBuffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time())
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Failed to get transform: {e}")
            return None
    
    def current_pose_callback(self, msg):
        self.observation_msg = msg
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z

        x_ori = msg.pose.orientation.x
        y_ori = msg.pose.orientation.y
        z_ori = msg.pose.orientation.z
        w_ori = msg.pose.orientation.w

        quat = [w_ori, x_ori, y_ori, z_ori]
        euler_angle = t3d.euler.quat2euler(quat)

        # self.observation_pose = [x_pos, y_pos, z_pos, x_ori, y_ori, z_ori, w_ori]
        self.observation_pose = [x_pos, y_pos, z_pos, euler_angle[0], euler_angle[1], euler_angle[2]]
        self.observation_current_pose = msg

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = cv2.resize(cv_image, (224, 224))
        self.image = cv_image
        self.frames.append(self.image)
        image_name = "/home/marija/exp-lerobot/examples/outputs/rollout_images/" + str(self.counter) + ".jpg"
        cv2.imwrite(image_name, self.image)
        self.counter +=1
          
    def publish_cmd_vel(self):

        if self.observation_pose is None:
            return
        
        if self.image is None:
            return

        base_gripper_tf = self.get_transform('link_base', 'gripper_base_link')
        if base_gripper_tf is None:
            return
        
        self.get_logger().info(f'========================={self.state}=========================')
        # if self.state == OperationState.INFERENCE:
        #     # self.get_logger().info(f'{self.state}')
        #     base_gripper = ros2numpy(base_gripper_tf.transform)
        #     # self.get_logger().info(f'{self.current_pose_relativ.pose}')
        #     # gripper_target = ros2numpy(self.current_pose_relativ.pose)

        #     # target_transform =  base_gripper @ gripper_target
        #     current_pose_target = ros2numpy(self.current_pose_relativ.pose)
        #     # target_transform = ros2numpy(self.observation_current_pose.pose) @ current_pose_target

        #     target_rotation = ros2numpy(self.observation_current_pose.pose) @ current_pose_target
        #     target_rotation = target_rotation[:3, :3]

        #     target_transform = ros2numpy(self.start_pose.pose) @ current_pose_target

        #     self.plotting_actions.append(target_transform[:3, 3])
        #     self.plotting_observations.append(self.observation_pose[:3])

        #     target_transform[:3, :3] = target_rotation

        #     pose = numpy2ros(target_transform, Pose)
        #     self.start_pose.pose = pose
        #     self.current_pose.pose = pose

        self.current_pose.header.stamp = self.get_clock().now().to_msg()
        self.current_pose.header.frame_id = 'link_base'

    
        if self.state == OperationState.IDLE:
            
            self.start_pose = self.observation_current_pose
            if self.start_pose  is not None:
                self.state = OperationState.INFERENCE
                self.timer = time.time()

        elif self.state == OperationState.INFERENCE:
            # Prepare observation for the policy running in Pytorch
            state = torch.from_numpy(np.array(self.observation_pose))
            state = state.to(torch.float32).to(self.device, non_blocking=True).unsqueeze(0)
            image = torch.from_numpy(self.image).to(torch.float32).permute(2, 0, 1).to(self.device, non_blocking=True).unsqueeze(0) / 255

            # Create the policy input dictionary
            observation = {
                "observation.state": state,
                "observation.image": image,
            }

            # Predict the next action with respect to the current observation
            tick = time.time()
            with torch.inference_mode():
                self.action = self.policy.select_action(observation).squeeze(0).cpu().numpy()
            tock = time.time()
            if self.action is None:
                return
            
            self.get_logger().info(f"Step {self.step}, Action: {self.action}")

            self.current_pose_relativ.header.stamp = self.get_clock().now().to_msg()
            self.current_pose_relativ.header.frame_id = 'gripper_link_base'
            self.current_pose_relativ.pose.position.x = (float(self.action[0]) / 1.5)
            self.current_pose_relativ.pose.position.y = (float(self.action[1]) / 1.0) 
            self.current_pose_relativ.pose.position.z = (float(self.action[2]) / 2.5)

            quat = t3d.euler.euler2quat((self.action[3] / 1.5), (self.action[4] / 6.5), (self.action[5] / 1.5))

            self.current_pose_relativ.pose.orientation.x = float(quat[1]) #0.0
            self.current_pose_relativ.pose.orientation.y = float(quat[2]) #0.0
            self.current_pose_relativ.pose.orientation.z = float(quat[3]) #0.0
            self.current_pose_relativ.pose.orientation.w = float(quat[0]) #1.0


            current_pose_target = ros2numpy(self.current_pose_relativ.pose)
            target_rotation = ros2numpy(self.observation_current_pose.pose) @ current_pose_target
            target_rotation = target_rotation[:3, :3]

            target_transform = ros2numpy(self.observation_current_pose.pose) @ current_pose_target
            # print(target_transform)

            self.plotting_actions.append(target_transform[:3, 3])
            self.plotting_observations.append(self.observation_pose[:3])

            target_transform[:3, :3] = target_rotation

            pose = numpy2ros(target_transform, Pose)
            self.start_pose.pose = pose
            self.current_pose.pose = pose

            if self.observation_msg.pose.position.z < 0.12: # 0.12 regular_value for picking
                self.state = OperationState.OPEN_GRIPPER
                self.timer = time.time()
                # time.sleep(15)
                self.get_logger().info('___________________________________ END ___________________________________')



            self.publisher_speed_limiter.publish(self.current_pose)

            self.step += 1

            if self.step > self.max_episode_steps:
                plot_action_trajectory(self.sim, self.plotting_observations, self.plotting_actions)
                self.state = OperationState.OPEN_GRIPPER
                self.timer = time.time()

        elif self.state == OperationState.OPEN_GRIPPER:
            # open later because make noise
            if not self.sim:
                msg = Int32()
                msg.data = 1
                self.switcher_publisher.publish(msg)

            if time.time() - self.timer > 1.5:
                self.state = OperationState.GO_CLOSE
                self.timer = time.time()

        elif self.state == OperationState.GO_CLOSE:
                self.observation_msg.pose.position.z = 0.085 # 0.085
                self.publisher_speed_limiter.publish(self.observation_msg)
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

            if time.time() -  self.timer > 1.5:
                self.state = OperationState.PICK_UP
                self.timer = time.time()
        elif self.state == OperationState.PICK_UP:
                self.observation_msg.pose.position.z = 0.29
                self.publisher_speed_limiter.publish(self.observation_msg)
                if time.time() - self.timer > 1.5:
                    self.state = OperationState.END
                    self.timer = time.time()

        elif self.state == OperationState.END:

            if time.time() - self.timer > 2.5:
                if not self.sim:
                    # shutdown gripper
                    msg = Int32()
                    msg.data = 2
                    self.switcher_publisher.publish(msg)

                self.save_video()

                self.get_logger().info(f"Finished publishing. Shutting down node...")
                rclpy.shutdown()
                exit(0)

       
        if check_pose_stamped_values(self.current_pose_relativ):
            return

    def save_video(self):
        # video_path = self.output_directory / "rollout_new_long_hz_model.mp4"
        # imageio.mimsave(str(video_path), numpy.stack(self.frames), fps=30)
        # self.get_logger().info(f"Video saved at {video_path}")
        pass

def main(args=None):

    import argparse
    parser = argparse.ArgumentParser(description='CmdVelPublisher')
    parser.add_argument('--sim', action='store_true', help='Use simulation')
    parsed_args = parser.parse_args()

    rclpy.init(args=args)
    cmd_vel_publisher = CmdVelPublisher(parsed_args.sim)
    rclpy.spin(cmd_vel_publisher)
    cmd_vel_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
