import rospy
import std_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from threading import Lock
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2

import cv2
import math
import os
import time
import torch
import numpy as np
import argparse
import json
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge

from config.config import cfg
from control_msg import PositionCommand
from policy.yopo_network import YopoNetwork
from policy.trt_runner import TensorRTPolicy, CachedTensorRTPolicy, planning_outputs_for_ros
from policy.rgb_feature_cache import RGBFeatureCache
from policy.trajectory_score import ClosedFormTrajectoryScore, choose_candidate
from policy.depth_expert_callback import depth_planning_enabled, run_depth_expert_frame
from policy.engine_metadata import read_metadata, validate_runtime_config, file_sha256
from policy.poly_solver import *
from policy.state_transform import *


class YopoNet:
    def __init__(self, config, weight):
        self.config = config
        rospy.init_node('yopo_net', anonymous=False)
        # load params
        cfg["train"] = False
        self.height = cfg['image_height']
        self.width = cfg['image_width']
        self.goal = np.array(self.config['goal'])
        self.auto_arrival_hold = bool(self.config.get('auto_arrival_hold', True))
        self.arrival_tolerance = float(self.config.get('arrival_tolerance', 5.0))
        self.goal_height = self.config.get('goal_height', 2.0)
        self.plan_from_reference = self.config['plan_from_reference']
        self.use_trt = self.config['use_tensorrt']
        self.use_feature_cache = bool(self.config.get('use_feature_cache', False))
        self.rgb_feature_cache = None
        self.verbose = self.config['verbose']
        self.visualize = self.config['visualize']
        self.publish_r2d_depth = bool(self.config.get("publish_r2d_depth", False))
        self.r2d_depth_topic = self.config.get("r2d_depth_topic", "/yopo_net/r2d_depth")
        self.publish_feature_visualization = bool(self.config.get("publish_feature_visualization", False))
        self.feature_visual_topic = self.config.get("feature_visual_topic", "/yopo_net/planning_feature_visual")
        self.publish_depth_expert_traj = bool(self.config.get("publish_depth_expert_traj", False))
        self.depth_expert_traj_topic = self.config.get("depth_expert_traj_topic", "/yopo_net/depth_expert_best_traj_visual")
        self.rgb_branch_traj_topic = self.config.get("rgb_branch_traj_topic", "/yopo_net/rgb_best_traj_visual")
        self.control_policy = self.config.get("control_policy", "rgb")
        self.selection_strategy = self.config.get("selection_strategy", "top8-cost")
        self.validation_mode = bool(self.config.get("validation_mode", False))
        self.depth_scale = float(self.config.get("depth_scale", 1.0))
        self.max_depth = float(self.config.get("max_depth", 20.0))
        self.Rotation_bc = R.from_euler('ZYX', [0, self.config['pitch_angle_deg'], 0], degrees=True).as_matrix()
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        # variables
        self.odom = Odometry()
        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = Lock()
        self.last_control_msg = None
        self.rgb_lock = Lock()
        self.rgb_buffer = []
        self.rgb_frame_ids = []
        self.rgb_frame_counter = 0
        self.pose_buffer = []
        self.latest_depth = None
        self.depth_lock = Lock()
        self.depth_inference_lock = Lock()
        self.Rotation_wb = np.eye(3, dtype=np.float32)
        self.Rotation_wc = np.eye(3, dtype=np.float32)
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.traj_time = self.lattice_primitive.segment_time
        self.closed_form_score = ClosedFormTrajectoryScore(self.traj_time)

        # eval
        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0
        self.image_fps = 30  # used only as processing time tolerance for printing logs
        # Normal testing starts immediately. Validation mode delegates start/stop
        # control to manual_simulation_eval via /yopo_net/flight_enable.
        self.flight_enabled = not self.validation_mode

        # Load Network
        self.depth_expert_policy = None
        if self.use_trt:
            if self.publish_r2d_depth or self.publish_feature_visualization:
                rospy.logwarn("R2D depth/feature publishing is disabled for TensorRT because the engine exports planning outputs only.")
                self.publish_r2d_depth = False
                self.publish_feature_visualization = False
            if self.use_feature_cache:
                self.policy = CachedTensorRTPolicy(
                    self.config["encoder_engine"], self.config["planner_engine"],
                    device=self.device, max_frames=cfg["data_timesteps"],
                )
            else:
                self.policy = TensorRTPolicy(weight, device=self.device)
        else:
            state_dict = torch.load(weight, map_location=self.device, weights_only=True)
            self.policy = YopoNetwork(
                rgb_pretrained=False,
                temporal_no_pose=YopoNetwork.checkpoint_temporal_no_pose(state_dict),
                disable_auxiliary_depth=YopoNetwork.checkpoint_auxiliary_depth_disabled(state_dict),
                score_mode=YopoNetwork.checkpoint_score_mode(state_dict),
                candidate_topk=YopoNetwork.checkpoint_candidate_topk(state_dict),
                single_frame_no_pose=YopoNetwork.checkpoint_single_frame(state_dict),
                rgb_encoder=YopoNetwork.checkpoint_rgb_encoder(state_dict),
                cnn_early_fusion=YopoNetwork.checkpoint_cnn_early_fusion(state_dict),
            )
            if self.policy.single_frame_no_pose:
                self.use_feature_cache = False
                self.publish_r2d_depth = False
                self.publish_feature_visualization = False
            load_result = self.policy.load_test_state_dict(state_dict)
            self.validate_visualization_weights(load_result)
            if load_result.unexpected_keys:
                print("Unexpected keys when loading test weights:", load_result.unexpected_keys)
            self.policy = self.policy.to(self.device)
            self.policy.eval()
            if self.use_feature_cache:
                if self.policy.single_frame_no_pose:
                    raise ValueError("Feature caching requires the two-frame temporal RGB model")
                self.rgb_feature_cache = RGBFeatureCache(
                    self.policy.encode_rgb_frame, max_frames=cfg["data_timesteps"]
                )
        if self.control_policy == "depth":
            self.publish_depth_expert_traj = True
        if self.publish_depth_expert_traj:
            self.depth_expert_policy = self.load_depth_expert_policy(self.config["depth_expert_weight"])
        self.warm_up()

        # ros publisher
        self.lattice_traj_pub = rospy.Publisher("/yopo_net/lattice_trajs_visual", PointCloud2, queue_size=1)
        self.best_traj_pub = rospy.Publisher("/yopo_net/best_traj_visual", PointCloud2, queue_size=1)
        self.all_trajs_pub = rospy.Publisher("/yopo_net/trajs_visual", PointCloud2, queue_size=1)
        self.ctrl_pub = rospy.Publisher(self.config["ctrl_topic"], PositionCommand, queue_size=1)
        self.r2d_depth_pub = None
        if self.publish_r2d_depth:
            self.r2d_depth_pub = rospy.Publisher(self.r2d_depth_topic, Image, queue_size=1)
        self.feature_visual_pub = None
        if self.publish_feature_visualization:
            self.feature_visual_pub = rospy.Publisher(self.feature_visual_topic, Image, queue_size=1)
        self.depth_expert_traj_pub = None
        if self.publish_depth_expert_traj:
            self.depth_expert_traj_pub = rospy.Publisher(self.depth_expert_traj_topic, PointCloud2, queue_size=1)
        self.rgb_branch_traj_pub = rospy.Publisher(self.rgb_branch_traj_topic, PointCloud2, queue_size=1)
        # ros subscriber
        self.odom_sub = rospy.Subscriber(self.config['odom_topic'], Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True)
        self.rgb_sub = rospy.Subscriber(self.config['rgb_topic'], Image, self.callback_rgb, queue_size=1, tcp_nodelay=True,buff_size=1 << 20)
        self.depth_sub = None
        if self.publish_feature_visualization or self.publish_depth_expert_traj:
            self.depth_sub = rospy.Subscriber(self.config['depth_topic'], Image, self.callback_depth, queue_size=1, tcp_nodelay=True)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1)
        self.selection_mode_pub = rospy.Publisher('/yopo_net/selection_mode', std_msgs.msg.String, queue_size=1, latch=True)
        self.selection_mode_pub.publish(std_msgs.msg.String(data=json.dumps({
            'selection_strategy': self.selection_strategy, 'control_policy': self.control_policy,
            'use_tensorrt': bool(self.use_trt), 'source_weight': weight})))
        rospy.loginfo('Trajectory selection strategy: %s', self.selection_strategy)
        self.flight_enable_sub = None
        if self.validation_mode:
            self.flight_enable_sub = rospy.Subscriber(
                "/yopo_net/flight_enable", std_msgs.msg.Bool,
                self.callback_flight_enable, queue_size=1,
            )
            rospy.loginfo("Validation mode enabled; waiting for /yopo_net/flight_enable")
        else:
            rospy.loginfo("Normal test mode enabled; YOPO planning starts without the validation program")
        # ros timer
        rospy.sleep(1.0)  # wait connection...
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        print("YOPO Net Node Ready!")
        rospy.spin()

    def callback_set_goal(self, data):
        self.goal = np.asarray([
            data.pose.position.x,
            data.pose.position.y,
            data.pose.position.z if self.goal_height is None else self.goal_height,
        ])
        self.arrive = False
        print(
            f"New Goal: ({data.pose.position.x:.1f}, "
            f"{data.pose.position.y:.1f}, {data.pose.position.z:.1f})"
        )

    def callback_flight_enable(self, data):
        with self.rgb_lock:
            return self._callback_flight_enable_locked(data)

    def _callback_flight_enable_locked(self, data):
        enabled = bool(data.data)
        if enabled == self.flight_enabled:
            return
        self.flight_enabled = enabled
        self.arrive = False
        self.ctrl_time = None
        self.rgb_buffer.clear()
        self.pose_buffer.clear()
        self.rgb_frame_ids.clear()
        if self.rgb_feature_cache is not None:
            self.rgb_feature_cache.clear()
        elif self.use_trt and self.use_feature_cache:
            self.policy.clear_cache()
        if enabled:
            self.desire_init = False
            # Initialize synchronously: an image can arrive before the next
            # odometry callback after a reset, otherwise it sees the previous
            # trial's reference position/acceleration for its first inference.
            if self.odom_init:
                position = self.odom.pose.pose.position
                velocity = self.odom.twist.twist.linear
                quaternion = self.odom.pose.pose.orientation
                self.desire_pos = np.array([position.x, position.y, position.z])
                self.desire_vel = np.array([velocity.x, velocity.y, velocity.z])
                self.desire_acc = np.zeros(3)
                self.last_yaw = R.from_quat([
                    quaternion.x, quaternion.y, quaternion.z, quaternion.w
                ]).as_euler('ZYX', degrees=False)[0]
            rospy.loginfo("YOPO planning enabled by /yopo_net/flight_enable")
        else:
            rospy.loginfo("YOPO planning disabled by /yopo_net/flight_enable")

    def validate_visualization_weights(self, load_result):
        if not (self.publish_r2d_depth or self.publish_feature_visualization):
            return

        required_prefixes = []
        if self.publish_r2d_depth or self.publish_feature_visualization:
            required_prefixes.append("auxiliary_depth_head.")
        if self.publish_feature_visualization:
            required_prefixes.extend(["depth_backbone.", "depth_plan_conv."])
        missing_required = [
            key for key in load_result.missing_keys
            if any(key.startswith(prefix) for prefix in required_prefixes)
        ]
        if missing_required:
            raise RuntimeError(
                "R2D visualization requires full weights. Use epoch*_all.pth instead of epoch*.pth. "
                f"Missing examples: {missing_required[:8]}"
            )

    def load_depth_expert_policy(self, weight):
        rospy.loginfo(f"Load depth expert visualization weight from: {weight}")
        state_dict = torch.load(weight, map_location=self.device, weights_only=True)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        policy = YopoNetwork(rgb_pretrained=False, score_mode=YopoNetwork.checkpoint_score_mode(state_dict),
                             candidate_topk=YopoNetwork.checkpoint_candidate_topk(state_dict)).to(self.device)
        load_result = policy.load_depth_expert_state_dict(state_dict)
        required_missing = [
            key for key in load_result.missing_keys
            if key.startswith(YopoNetwork.DEPTH_EXPERT_STATE_PREFIXES)
        ]
        if required_missing:
            raise RuntimeError(
                "Depth expert trajectory must use epoch*_depth_expert.pth, or a full checkpoint saved at depth pretrain. "
                f"Missing examples: {required_missing[:8]}"
            )
        if load_result.unexpected_keys:
            print("Unexpected keys when loading depth expert visualization weights:", load_result.unexpected_keys)
        policy.eval()
        return policy

    # the first frame
    def callback_odometry(self, data):
        self.odom = data
        if not self.desire_init:
            self.desire_pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            self.desire_vel = np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            #实物加速度一直用的是参考加速度
            ypr = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                               self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_euler('ZYX', degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True

        pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        if (
            self.auto_arrival_hold
            and np.linalg.norm(pos - self.goal) < self.arrival_tolerance
            and not self.arrive
        ):
            print("Arrive!")
            self.arrive = True
            hover_msg = PositionCommand()
            hover_msg.header.stamp = rospy.Time.now()
            hover_msg.header.frame_id = data.header.frame_id
            hover_msg.trajectory_flag = hover_msg.TRAJECTORY_STATUS_EMPTY
            hover_msg.position.x = float(pos[0])
            hover_msg.position.y = float(pos[1])
            hover_msg.position.z = float(pos[2])
            hover_msg.velocity.x = 0.0
            hover_msg.velocity.y = 0.0
            hover_msg.velocity.z = 0.0
            hover_msg.acceleration.x = 0.0
            hover_msg.acceleration.y = 0.0
            hover_msg.acceleration.z = 0.0
            hover_msg.yaw = self.last_yaw
            hover_msg.yaw_dot = 0.0
            with self.lock:
                self.desire_pos = pos.copy()
                self.desire_vel = np.zeros(3)
                self.desire_acc = np.zeros(3)
                self.last_control_msg = hover_msg
            # Immediately replace the last forward trajectory command instead
            # of waiting for the next control timer tick.
            self.ctrl_pub.publish(hover_msg)

    def process_odom(self):
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        Rotation_cw = self.Rotation_wc.T

        # vel and acc
        vel_w = self.desire_vel if self.plan_from_reference else np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
        vel_b = np.dot(Rotation_cw, vel_w)
        acc_w = self.desire_acc
        acc_b = np.dot(Rotation_cw, acc_w)
        # goal_dir
        goal_w = self.goal - self.desire_pos
        goal_b = np.dot(Rotation_cw, goal_w)

        return np.concatenate((vel_b, acc_b, goal_b), axis=0).astype(np.float32)

    def build_pose_from_odom(self):
        quat_xyzw = np.array([
            self.odom.pose.pose.orientation.x,
            self.odom.pose.pose.orientation.y,
            self.odom.pose.pose.orientation.z,
            self.odom.pose.pose.orientation.w,
        ], dtype=np.float32)
        pos_w = np.array([
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            self.odom.pose.pose.position.z,
        ], dtype=np.float32)
        return np.concatenate((quat_xyzw, pos_w), axis=0)

    @staticmethod
    def build_relative_pose(prev_pose, cur_pose):
        q_rel = (R.from_quat(cur_pose[0:4]).inv() * R.from_quat(prev_pose[0:4])).as_quat()
        t_rel = R.from_quat(cur_pose[0:4]).inv().apply(prev_pose[4:7] - cur_pose[4:7])
        return np.concatenate((q_rel, t_rel), axis=0).astype(np.float32)

    def process_rgb(self, data):
        rgb_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        if rgb_image.shape[0] != self.height or rgb_image.shape[1] != self.width:
            rgb_image = cv2.resize(
                rgb_image, (self.width, self.height), interpolation=cv2.INTER_AREA
            )

        #和训练一样的归一化  
        rgb_image = rgb_image.astype(np.float32) / 255.0
        #mean = np.array([0.37596039, 0.43150085, 0.42616264], dtype=np.float32)
        #std = np.array([0.14942774, 0.13605953, 0.14103027], dtype=np.float32)
        #rgb_image = (rgb_image - mean) / std
        return rgb_image.astype(np.float32)

    def process_depth(self, data):
        depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        depth_image = np.asarray(depth_image)
        if depth_image.ndim == 3:
            depth_image = depth_image[:, :, 0]

        if depth_image.dtype == np.uint16:
            # ROS uint16 depth is millimeters; match training's depth / max_depth.
            depth_image = depth_image.astype(np.float32) / 1000.0
            if self.max_depth > 0:
                depth_image = depth_image / self.max_depth
        elif depth_image.dtype == np.uint8:
            depth_image = depth_image.astype(np.float32) / 255.0
        else:
            depth_image = depth_image.astype(np.float32)
            depth_image = np.nan_to_num(depth_image, nan=self.max_depth, posinf=self.max_depth, neginf=0.0)
            if self.depth_scale > 0:
                depth_image = depth_image * self.depth_scale
            if self.max_depth > 0:
                depth_image = depth_image / self.max_depth

        depth_image = np.clip(depth_image, 0.0, 1.0)
        if depth_image.shape[0] != self.height or depth_image.shape[1] != self.width:
            depth_image = cv2.resize(depth_image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return depth_image.astype(np.float32)

    @torch.inference_mode()
    def callback_depth(self, data):
        depth = self.process_depth(data)
        with self.depth_lock:
            self.latest_depth = depth
        if not (self.publish_depth_expert_traj or self.control_policy == "depth"):
            return
        # Do not queue concurrent depth publishers behind a running inference.
        if not self.depth_inference_lock.acquire(blocking=False):
            return
        try:
            if not depth_planning_enabled(self):
                return
            odom_msg = self._get_depth_odom_for_message(data)
            if odom_msg is None:
                return
            started = time.time()
            result = run_depth_expert_frame(self, depth, odom_msg)
            if result is None:
                return
            score, endstate_w, action_id = result["score"], result["endstate_w"], result["action_id"]
            snapshot = result["snapshot"]
            self.publish_depth_expert_trajectory(
                score, endstate_w, action_id,
                start_pos=snapshot["start_pos"], start_vel=snapshot["start_vel"],
                start_acc=snapshot["start_acc"],
            )
            if self.control_policy == "depth":
                self.visualize_trajectory(score, endstate_w, **snapshot)
            if self.verbose:
                rospy.loginfo_throttle(
                    0.5, f"Depth callback planning: {(time.time() - started) * 1000.0:.2f} ms"
                )
        finally:
            self.depth_inference_lock.release()

    def _get_depth_odom_for_message(self, data):
        # The fixed-goal node has no odometry history; snapshot the latest message.
        return self.odom

    @torch.inference_mode()
    def callback_rgb(self, data):
        with self.rgb_lock:
            return self._callback_rgb_locked(data)

    def _callback_rgb_locked(self, data):
        if self.control_policy == "depth" and not (
            self.publish_r2d_depth or self.publish_feature_visualization
            or self.rgb_branch_traj_pub.get_num_connections() > 0
        ):
            return
        if not self.flight_enabled or not self.odom_init or self.arrive:
            return

        # 1. RGB Image Process
        time0 = time.time()
        rgb = self.process_rgb(data)
        pose = self.build_pose_from_odom()
        self.rgb_frame_counter += 1
        self.rgb_frame_ids.append(self.rgb_frame_counter)
        self.rgb_buffer.append(rgb)
        self.pose_buffer.append(pose)
        if len(self.rgb_buffer) > cfg["data_timesteps"]:
            self.rgb_buffer.pop(0)
            self.rgb_frame_ids.pop(0)
            self.pose_buffer.pop(0)
        if len(self.rgb_buffer) < cfg["data_timesteps"]:
            self.rgb_buffer.insert(0, rgb.copy())
            self.rgb_frame_ids.insert(0, self.rgb_frame_counter)
            self.pose_buffer.insert(0, pose.copy())

        # 2. YOPO Network Inference
        # input prepare
        time1 = time.time()
        rgb_input = np.stack(self.rgb_buffer, axis=0).transpose(0, 3, 1, 2)[None, ...]
        no_pose = not self.use_trt and (self.policy.single_frame_no_pose or self.policy.temporal_no_pose)
        pose_input = None if no_pose else self.build_relative_pose(self.pose_buffer[0], self.pose_buffer[-1])[None, :]
        obs_input = self.process_odom()[None, :]
        depth_input = self.get_latest_depth_input()
        rgb_input = torch.from_numpy(rgb_input).to(self.device, non_blocking=True)
        if pose_input is not None:
            pose_input = torch.from_numpy(pose_input).to(self.device, non_blocking=True)
        obs_input = torch.from_numpy(obs_input).to(self.device, non_blocking=True)
        if depth_input is not None:
            depth_input = torch.from_numpy(depth_input).to(self.device, non_blocking=True)
        # torch.cuda.synchronize()

        time2 = time.time()
        policy_outputs = self.forward_policy(rgb_input, obs_input, pose_input, depth_input, frame_ids=tuple(self.rgb_frame_ids))
        endstate_pred, selection_logits = policy_outputs[:2]
        depth_est = policy_outputs[2] if len(policy_outputs) > 2 else None
        depth_feature = policy_outputs[3] if len(policy_outputs) > 3 else None
        r2d_feature = policy_outputs[4] if len(policy_outputs) > 4 else None
        safety_score = policy_outputs[5] if len(policy_outputs) > 5 else None
        endstate_pred = endstate_pred[0]
        selection_logits = selection_logits[0]
        if self.device == "cuda":
            torch.cuda.synchronize()
        time3 = time.time()

        start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))

        rgb_score, rgb_endstate_w, rgb_action_id = self.select_action_from_outputs(
            endstate_pred,
            selection_logits,
            safety_score=safety_score[0] if safety_score is not None else None,
        )
        if self.control_policy == "rgb":
            with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety
                self.optimal_poly_x = Poly5Solver(start_pos[0], start_vel[0], self.desire_acc[0], rgb_endstate_w[rgb_action_id, 0, 0] + start_pos[0],
                                                  rgb_endstate_w[rgb_action_id, 0, 1], rgb_endstate_w[rgb_action_id, 0, 2], self.traj_time)
                self.optimal_poly_y = Poly5Solver(start_pos[1], start_vel[1], self.desire_acc[1], rgb_endstate_w[rgb_action_id, 1, 0] + start_pos[1],
                                                  rgb_endstate_w[rgb_action_id, 1, 1], rgb_endstate_w[rgb_action_id, 1, 2], self.traj_time)
                self.optimal_poly_z = Poly5Solver(start_pos[2], start_vel[2], self.desire_acc[2], rgb_endstate_w[rgb_action_id, 2, 0] + start_pos[2],
                                                  rgb_endstate_w[rgb_action_id, 2, 1], rgb_endstate_w[rgb_action_id, 2, 2], self.traj_time)
                self.ctrl_time = 0.0
        time4 = time.time()
        self.publish_r2d_depth_image(depth_est, data.header)
        self.publish_feature_visual_image(depth_feature, r2d_feature, data.header)
        self.publish_rgb_branch_trajectory(start_pos, start_vel, rgb_endstate_w, rgb_action_id)
        if self.control_policy == "rgb":
            self.visualize_trajectory(rgb_score, rgb_endstate_w)
        time5 = time.time()

        self.print_time(time0, time1, time2, time3, time4, time5)

    def control_pub(self, _timer):
        if not self.flight_enabled:
            return
        if self.arrive and self.last_control_msg is not None:
            with self.lock:
                self.last_control_msg.header.stamp = rospy.Time.now()
                self.ctrl_pub.publish(self.last_control_msg)
            return
        if self.ctrl_time is None or self.ctrl_time > self.traj_time:
            return

        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety and publish frequency
            self.ctrl_time += self.ctrl_dt
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
            control_msg.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
            control_msg.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
            control_msg.position.z = self.optimal_poly_z.get_position(self.ctrl_time)
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(self.ctrl_time)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(self.ctrl_time)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(self.ctrl_time)
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(self.ctrl_time)
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(self.ctrl_time)
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(self.ctrl_time)
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])
            goal_dir = self.goal - self.desire_pos
            yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.last_control_msg = control_msg
            self.ctrl_pub.publish(control_msg)

    def process_output(self, endstate_pred, selection_logits, return_all_preds=False):
        if getattr(self.policy, 'score_mode', 'selection') != 'selection':
            raise RuntimeError('Collision costs require select_action_from_outputs with analytic costs')
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        selection_logits = selection_logits.reshape(self.lattice_primitive.traj_num)

        if not return_all_preds:
            action_id = np.argmax(selection_logits)
            endstate = endstate_pred[action_id, :][np.newaxis, :]
            score = selection_logits[action_id]
        else:
            score = selection_logits
            endstate = endstate_pred

        return endstate, score

    def visualize_trajectory(self, pred_score, pred_endstate,
                             start_pos=None, start_vel=None, start_acc=None, rotation_wc=None):
        dt = self.traj_time / 20.0
        if start_pos is None:
            start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        if start_vel is None:
            start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
        start_acc = self.desire_acc if start_acc is None else start_acc
        rotation_wc = self.Rotation_wc if rotation_wc is None else rotation_wc
        # best predicted trajectory
        if self.best_traj_pub.get_num_connections() > 0:
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                self.optimal_poly_x.get_position(t_values),
                self.optimal_poly_y.get_position(t_values),
                self.optimal_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.best_traj_pub.publish(point_cloud_msg)
        # lattice primitive
        if self.visualize and self.lattice_traj_pub.get_num_connections() > 0:
            lattice_endstate = self.lattice_primitive.lattice_pos_node.cpu().numpy()
            lattice_endstate = np.dot(lattice_endstate, rotation_wc.T)
            zero_state = np.zeros_like(lattice_endstate)
            lattice_poly_x = Polys5Solver(start_pos[0], start_vel[0], start_acc[0],
                                          lattice_endstate[:, 0] + start_pos[0], zero_state[:, 0], zero_state[:, 0], self.traj_time)
            lattice_poly_y = Polys5Solver(start_pos[1], start_vel[1], start_acc[1],
                                          lattice_endstate[:, 1] + start_pos[1], zero_state[:, 1], zero_state[:, 1], self.traj_time)
            lattice_poly_z = Polys5Solver(start_pos[2], start_vel[2], start_acc[2],
                                          lattice_endstate[:, 2] + start_pos[2], zero_state[:, 2], zero_state[:, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                lattice_poly_x.get_position(t_values),
                lattice_poly_y.get_position(t_values),
                lattice_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.lattice_traj_pub.publish(point_cloud_msg)
        # all predicted trajectories
        if self.visualize and self.all_trajs_pub.get_num_connections() > 0:
            all_poly_x = Polys5Solver(start_pos[0], start_vel[0], start_acc[0],
                                      pred_endstate[:, 0, 0] + start_pos[0], pred_endstate[:, 0, 1], pred_endstate[:, 0, 2], self.traj_time)
            all_poly_y = Polys5Solver(start_pos[1], start_vel[1], start_acc[1],
                                      pred_endstate[:, 1, 0] + start_pos[1], pred_endstate[:, 1, 1], pred_endstate[:, 1, 2], self.traj_time)
            all_poly_z = Polys5Solver(start_pos[2], start_vel[2], start_acc[2],
                                      pred_endstate[:, 2, 0] + start_pos[2], pred_endstate[:, 2, 1], pred_endstate[:, 2, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                all_poly_x.get_position(t_values),
                all_poly_y.get_position(t_values),
                all_poly_z.get_position(t_values)
            ), axis=-1)
            scores = np.repeat(pred_score, t_values.size)
            points_array = np.column_stack((points_array, scores))
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1), PointField('intensity', 12, PointField.FLOAT32, 1)]
            point_cloud_msg = point_cloud2.create_cloud(header, fields, points_array)
            self.all_trajs_pub.publish(point_cloud_msg)

    def print_time(self, time0, time1, time2, time3, time4, time5):
        """
        Performance reference: PyTorch model should take < 5 ms; TensorRT model should take < 1 ms

        Notes:
        - Running program and enabling RViz under WSL greatly increase processing time, and Ubuntu does not have these issues
        - Even with queue_size=1, it may cause message accumulation and lag when processing time exceeds the image frequency
        """
        self.time_interpolation = self.time_interpolation + (time1 - time0)
        self.time_prepare = self.time_prepare + (time2 - time1)
        self.time_forward = self.time_forward + (time3 - time2)
        self.time_process = self.time_process + (time4 - time3)
        self.time_visualize = self.time_visualize + (time5 - time4)
        self.count = self.count + 1

        total_time = (time5 - time0) * 1000
        tolerance = 1000.0 / self.image_fps
        if total_time > tolerance:
            rospy.logwarn(f"Warn: Processing time {(time5 - time0) * 1000:.2f} ms exceeds {tolerance:.2f} ms, may cause message lag!")
            print(f"\033[34mCurrent Time Consuming:\033[0m "
                  f"rgb-process: \033[32m{1000 * (time1 - time0):.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * (time2 - time1):.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * (time3 - time2):.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * (time4 - time3):.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * (time5 - time4):.2f} ms\033[0m")
        if self.verbose or (total_time > tolerance):
            print(f"\033[34mAverage Time Consuming:\033[0m "
                  f"rgb-process: \033[32m{1000 * self.time_interpolation / self.count:.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * self.time_prepare / self.count:.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * self.time_forward / self.count:.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * self.time_process / self.count:.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * self.time_visualize / self.count:.2f} ms\033[0m")

    def warm_up(self):
        rgb = torch.zeros((1, cfg["data_timesteps"], 3, self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        pose = torch.zeros((1, 7), dtype=torch.float32, device=self.device)
        pose[:, 3] = 1.0
        _ = self.forward_policy(rgb, obs, pose)

    def get_latest_depth_input(self):
        if not (self.publish_feature_visualization or self.publish_depth_expert_traj):
            return None
        with self.depth_lock:
            if self.latest_depth is None:
                return None
            depth = self.latest_depth.copy()
        return depth[None, None, :, :]

    @torch.inference_mode()
    def forward_policy(self, rgb, obs, pose, depth=None, frame_ids=None):
        if self.use_trt:
            if self.use_feature_cache:
                outputs = self.policy(rgb, obs, pose, frame_ids=frame_ids)
            else:
                outputs = self.policy(rgb, obs, pose)
            return planning_outputs_for_ros(outputs)
        want_features = self.publish_feature_visualization and depth is not None
        if self.rgb_feature_cache is not None:
            features = self.rgb_feature_cache.encode_sequence(rgb, frame_ids)
            outputs = self.policy.inference_from_rgb_features(
                features, obs, pose, return_depth=self.publish_r2d_depth or want_features)
            depth_feature = None
            if want_features:
                depth_map = self.policy.depth_plan_conv(self.policy.depth_backbone(depth[:, -1:].contiguous()))
                depth_feature = depth_map.flatten(2).permute(0, 2, 1)
        else:
            mode = 'rgb_percepetion_distill' if want_features else 'test_r2d' if self.publish_r2d_depth else 'test'
            outputs = self.policy.inference(depth=depth, rgb=rgb, obs=obs.clone(), pose=pose, mode=mode)
            depth_feature = outputs.get('depth_feature')
        if self.publish_feature_visualization and depth is None:
            rospy.logwarn_throttle(1.0, "Waiting for depth image before publishing planning feature visualization.")
        return (outputs['endstate'], outputs['score'], outputs.get('depth_est'),
                depth_feature, outputs.get('r2d_feature'), outputs.get('safety_score'))

    def forward_depth_expert_policy(self, depth, obs):
        if depth is None:
            rospy.logwarn_throttle(1.0, "Depth expert is enabled, but no depth image has been received yet.")
            return None
        if self.depth_expert_policy is None:
            rospy.logwarn_throttle(1.0, "Depth expert policy is not loaded.")
            return None
        return self.depth_expert_policy.inference(depth=depth, obs=obs.clone(), mode="depth_pretrain")

    def select_action_from_outputs(self, endstate_pred, selection_logits, rotation_wc=None,
                                   safety_score=None, policy=None, start_pos=None,
                                   start_vel=None, start_acc=None, goal=None):
        policy = self.policy if policy is None else policy
        endstate = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T.detach().cpu().numpy()
        selection_score = selection_logits.reshape(self.lattice_primitive.traj_num).detach().cpu().numpy()

        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)
        rotation_wc = self.Rotation_wc if rotation_wc is None else rotation_wc
        endstate_w = np.matmul(rotation_wc.astype(endstate_c.dtype, copy=False), endstate_c)
        score = selection_score
        mode = getattr(policy, 'score_mode', 'selection')
        if getattr(self, 'selection_strategy', 'top8-cost') == 'classification-top1':
            if mode not in ('selection', 'topk-collision'):
                raise ValueError('Classification Top1 requires a classification-logit output')
            action_id = int(np.argmax(selection_score))
        elif mode != 'selection':
            if start_pos is None:
                start_pos = self.desire_pos if self.plan_from_reference else np.array([
                    self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z])
            if start_vel is None:
                start_vel = self.desire_vel if self.plan_from_reference else np.array([
                    self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
            if mode == 'topk-collision' and safety_score is None:
                raise ValueError('Two-stage selection requires safety_score')
            collision = safety_score.detach().cpu().numpy().reshape(-1) if mode == 'topk-collision' else selection_score
            score = self.closed_form_score.total_score(collision, endstate_w, start_pos, start_vel,
                                                      self.desire_acc if start_acc is None else start_acc,
                                                      self.goal if goal is None else goal)
            action_id = choose_candidate(score, selection_score if mode == 'topk-collision' else None,
                                         getattr(policy, 'candidate_topk', 8))
        else:
            action_id = int(np.argmax(selection_score))
        if not self.visualize:
            endstate_w = endstate_w[action_id:action_id + 1]
            score = score[action_id:action_id + 1]
            action_id = 0
        return score, endstate_w, action_id

    def publish_r2d_depth_image(self, depth_est, header):
        if depth_est is None or self.r2d_depth_pub is None or self.r2d_depth_pub.get_num_connections() == 0:
            return

        depth_image = depth_est[0].detach().float().cpu().numpy()
        # The R2D head predicts normalized depth. ROS 32FC1 depth images use
        # meters, so restore the training range before publishing.
        depth_image = np.clip(depth_image, 0.0, 1.0) * self.max_depth
        depth_msg = self.bridge.cv2_to_imgmsg(depth_image.astype(np.float32), encoding="32FC1")
        depth_msg.header = header
        self.r2d_depth_pub.publish(depth_msg)

    def publish_depth_expert_trajectory(self, score, endstate_w, action_id,
                                        start_pos=None, start_vel=None, start_acc=None):
        if (
            not self.publish_depth_expert_traj
            or self.depth_expert_traj_pub is None
            or score is None
            or endstate_w is None
            or action_id is None
        ):
            return
        if start_pos is None:
            start_pos = self.desire_pos if self.plan_from_reference else np.array((
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ))
        if start_vel is None:
            start_vel = self.desire_vel if self.plan_from_reference else np.array((
                self.odom.twist.twist.linear.x,
                self.odom.twist.twist.linear.y,
                self.odom.twist.twist.linear.z,
            ))
        self.publish_best_polyline(
            self.depth_expert_traj_pub,
            start_pos,
            start_vel,
            endstate_w[action_id],
            start_acc=start_acc,
        )

    def publish_rgb_branch_trajectory(self, start_pos, start_vel, endstate_w, action_id):
        if self.control_policy != "depth" or self.rgb_branch_traj_pub is None:
            return
        self.publish_best_polyline(
            self.rgb_branch_traj_pub,
            start_pos,
            start_vel,
            endstate_w[action_id],
        )

    def publish_best_polyline(self, publisher, start_pos, start_vel, endstate_w, start_acc=None):
        start_acc = self.desire_acc if start_acc is None else start_acc
        dt = self.traj_time / 20.0
        t_values = np.arange(0, self.traj_time, dt)
        poly_x = Poly5Solver(start_pos[0], start_vel[0], start_acc[0],
                             endstate_w[0, 0] + start_pos[0], endstate_w[0, 1], endstate_w[0, 2], self.traj_time)
        poly_y = Poly5Solver(start_pos[1], start_vel[1], start_acc[1],
                             endstate_w[1, 0] + start_pos[1], endstate_w[1, 1], endstate_w[1, 2], self.traj_time)
        poly_z = Poly5Solver(start_pos[2], start_vel[2], start_acc[2],
                             endstate_w[2, 0] + start_pos[2], endstate_w[2, 1], endstate_w[2, 2], self.traj_time)
        points_array = np.stack((
            poly_x.get_position(t_values),
            poly_y.get_position(t_values),
            poly_z.get_position(t_values),
        ), axis=-1)
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world'
        publisher.publish(point_cloud2.create_cloud_xyz32(header, points_array))

    def publish_feature_visual_image(self, depth_feature, r2d_feature, header):
        if (
            depth_feature is None
            or r2d_feature is None
            or self.feature_visual_pub is None
            or self.feature_visual_pub.get_num_connections() == 0
        ):
            return

        depth_map = self.feature_to_heatmap(depth_feature[0])
        r2d_map = self.feature_to_heatmap(r2d_feature[0])
        diff_map = self.normalize_image(np.abs(depth_map.astype(np.float32) - r2d_map.astype(np.float32)))
        visual = np.concatenate(
            (
                cv2.applyColorMap(depth_map, cv2.COLORMAP_VIRIDIS),
                cv2.applyColorMap(r2d_map, cv2.COLORMAP_VIRIDIS),
                cv2.applyColorMap(diff_map, cv2.COLORMAP_INFERNO),
            ),
            axis=1,
        )
        visual = cv2.resize(visual, (self.width, max(1, self.height // 3)), interpolation=cv2.INTER_NEAREST)
        feature_msg = self.bridge.cv2_to_imgmsg(visual, encoding="bgr8")
        feature_msg.header = header
        self.feature_visual_pub.publish(feature_msg)

    @staticmethod
    def normalize_image(image):
        image = image.astype(np.float32)
        min_value = float(np.min(image))
        max_value = float(np.max(image))
        if max_value - min_value < 1e-8:
            return np.zeros(image.shape, dtype=np.uint8)
        return ((image - min_value) * (255.0 / (max_value - min_value))).astype(np.uint8)

    def feature_to_heatmap(self, feature):
        feature = feature.detach().float().cpu().numpy()
        if feature.ndim != 2:
            raise ValueError(f"Expected feature shape [seq, channels], got {feature.shape}")

        seq_len = feature.shape[0]
        grid_h = max(1, int(round(math.sqrt(seq_len * self.height / self.width))))
        while grid_h > 1 and seq_len % grid_h != 0:
            grid_h -= 1
        grid_w = seq_len // grid_h
        heatmap = np.linalg.norm(feature, axis=1).reshape(grid_h, grid_w)
        return self.normalize_image(heatmap)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    parser.add_argument("--use_feature_cache", type=int, default=1, choices=[0, 1], help="reuse encoded history frames; TensorRT requires split engines")
    parser.add_argument("--encoder_engine", type=str, default="yopo_encoder.engine", help="cached single-frame RGB encoder engine")
    parser.add_argument("--planner_engine", type=str, default="yopo_planner.engine", help="cached feature-input temporal planner engine")
    parser.add_argument("--trial", type=int, default=116, help="trial number")
    parser.add_argument("--epoch", type=int, default=160, help="epoch number")
    parser.add_argument("--weight", type=str, default="", help="explicit checkpoint path")
    parser.add_argument("--training_config", type=str, default="", help="explicit saved ablation_config.json for checkpoint settings")
    parser.add_argument("--velocity", type=float, default=None, help="override inference speed in m/s after loading training configuration")
    parser.add_argument("--arrival_tolerance", type=float, default=5.0, help="distance to goal in meters for optional arrival hold")
    parser.add_argument("--goal_height", type=float, default=1.5, help="explicit goal height in world meters (legacy default: 2m)")
    parser.add_argument("--check_only", action="store_true", help="validate checkpoint metadata and print launch settings without starting ROS or flight")
    parser.add_argument("--publish_r2d_depth", type=int, default=0, choices=[0, 1], help="publish RGB-to-depth reconstruction during inference")
    parser.add_argument("--r2d_depth_topic", type=str, default="/yopo_net/r2d_depth", help="R2D reconstructed depth image topic")
    parser.add_argument("--publish_feature_visualization", type=int, default=0, choices=[0, 1], help="publish depth/RGB planning feature visualization")
    parser.add_argument("--feature_visual_topic", type=str, default="/yopo_net/planning_feature_visual", help="planning feature visualization topic")
    parser.add_argument("--publish_depth_expert_traj", type=int, default=0, choices=[0, 1], help="publish best trajectory selected by the depth expert branch")
    parser.add_argument("--depth_expert_traj_topic", type=str, default="/yopo_net/depth_expert_best_traj_visual", help="depth expert best trajectory topic")
    parser.add_argument("--rgb_branch_traj_topic", type=str, default="/yopo_net/rgb_best_traj_visual", help="RGB branch best trajectory topic when depth controls")
    parser.add_argument("--depth_expert_weight", type=str, default="saved/YOPO_90/epoch100_depth_expert.pth", help="explicit depth expert checkpoint path")
    parser.add_argument("--control_policy", type=str, default="rgb", choices=["rgb", "depth"], help="which branch controls the quadrotor")
    parser.add_argument("--selection_strategy", choices=["top8-cost", "classification-top1"], default="top8-cost",
                        help="default Top8 predicted-cost reranking, or highest classification logit only (no cost reranking)")
    parser.add_argument("--depth_topic", type=str, default="/depth_image", help="RGB-aligned front depth image topic for feature comparison")
    parser.add_argument("--max_depth", type=float, default=20.0, help="depth range in meters; normalizes 16UC1 millimeters or 32FC1 meters to [0,1]")
    parser.add_argument("--depth_scale", type=float, default=1.0, help="scale applied before max_depth normalization for float depth")
    parser.add_argument("--validation_mode", type=int, default=0, choices=[0, 1], help="wait for manual_simulation_eval to enable flight (default: 0, start normally without it)")
    parser.add_argument("--auto_arrival_hold", type=int, default=1, choices=[0, 1], help="stop and hover when this node reaches the current goal; set to 0 when manual_simulation_eval manages waypoints")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.weight:
        weight = args.weight
    elif args.use_tensorrt:
        weight = base_dir + "/yopo.engine"
    elif args.publish_r2d_depth or args.publish_feature_visualization:
        weight = base_dir + "/saved/YOPO_{}/epoch{}_all.pth".format(args.trial, args.epoch)
    else:
        weight = base_dir + "/saved/YOPO_{}/epoch{}.pth".format(args.trial, args.epoch)
    training_config = args.training_config
    if not training_config:
        # Checkpoints must use their saved architecture and cost settings.
        # Split TensorRT engines normally live outside the checkpoint folder.
        config_dir = (os.path.join(base_dir, "saved", "YOPO_{}".format(args.trial))
                      if args.use_tensorrt else os.path.dirname(os.path.abspath(weight)))
        candidate = os.path.join(config_dir, "ablation_config.json")
        if os.path.isfile(candidate):
            training_config = candidate
    if training_config:
        with open(training_config) as stream:
            cfg._data.update(json.load(stream))
        print("load training configuration from:", training_config)
    if args.velocity is not None:
        if not np.isfinite(args.velocity) or args.velocity <= 0:
            raise ValueError("velocity must be finite and positive")
        cfg['velocity'] = args.velocity
    if not np.isfinite(args.arrival_tolerance) or args.arrival_tolerance <= 0:
        raise ValueError("arrival_tolerance must be finite and positive")
    if not np.isfinite(args.goal_height):
        raise ValueError("goal_height must be finite")
    cfg['goal_length'] = 2.0 * cfg['radio_range']
    cfg['sgm_time'] = 2.0 * cfg['radio_range'] / cfg['vel_max_train']
    cfg['traj_num'] = cfg['horizon_num'] * cfg['vertical_num'] * cfg['radio_num']
    print("load weight from:", weight)
    depth_expert_weight = args.depth_expert_weight
    if args.publish_depth_expert_traj and not depth_expert_weight:
        depth_expert_weight = base_dir + "/saved/YOPO_{}/epoch{}_depth_expert.pth".format(args.trial, cfg["depth_pretrain_num"])

    settings = {'use_tensorrt': args.use_tensorrt,
                'use_feature_cache': args.use_feature_cache,
                'encoder_engine': args.encoder_engine,
                'planner_engine': args.planner_engine,
                'goal': [40, 0, 1.2],      # 目标点位置
                'validation_mode': args.validation_mode,
                'auto_arrival_hold': args.auto_arrival_hold,
                'arrival_tolerance': args.arrival_tolerance,
                'goal_height': args.goal_height,
                'pitch_angle_deg': 0,   # 相机俯仰角(仰为负)
                'odom_topic': '/sim/odom',
                #'odom_topic': '/vins_estimator/imu_propagate',                   # 里程计话题
                'rgb_topic': '/rgb/image',                   # 针孔 RGB 图话题
                'depth_topic': args.depth_topic,             # 深度图话题，用于特征对比
                'ctrl_topic': '/so3_control/pos_cmd',        # 控制器话题
                'plan_from_reference': False,   # 从参考状态规划？位置控制器: True, 神经网络直接控制: False
                'publish_r2d_depth': args.publish_r2d_depth,
                'r2d_depth_topic': args.r2d_depth_topic,
                'publish_feature_visualization': args.publish_feature_visualization,
                'feature_visual_topic': args.feature_visual_topic,
                'publish_depth_expert_traj': args.publish_depth_expert_traj,
                'depth_expert_traj_topic': args.depth_expert_traj_topic,
                'rgb_branch_traj_topic': args.rgb_branch_traj_topic,
                'depth_expert_weight': depth_expert_weight,
                'control_policy': args.control_policy,
                'selection_strategy': args.selection_strategy,
                'max_depth': args.max_depth,
                'depth_scale': args.depth_scale,
                'verbose': True,               # 打印耗时？
                'visualize': False              # 可视化所有轨迹？(实飞改为False节省计算)
                }
    deployment_metadata={}
    if args.use_tensorrt:
        sources=({'encoder':args.encoder_engine,'planner':args.planner_engine}
                 if args.use_feature_cache else {'full':weight})
        checkpoint_sha=file_sha256(args.weight) if args.weight and args.use_feature_cache else None
        for role,path in sources.items():
            info=read_metadata(path)
            if info is None:
                if checkpoint_sha:raise ValueError('Explicit checkpoint requires TensorRT metadata: '+path)
                continue
            validate_runtime_config(info,cfg)
            if info['role']!=role:raise ValueError('Wrong TensorRT component role: '+path)
            if checkpoint_sha and info['weight_sha256']!=checkpoint_sha:
                raise ValueError('TensorRT engine does not match requested checkpoint: '+path)
            deployment_metadata[role]={'path':path,'weight_sha256':info['weight_sha256'],'deployment_id':info['deployment_id'],'engine_sha256':info['engine_sha256']}
        if len(deployment_metadata)==2 and len({m['deployment_id'] for m in deployment_metadata.values()})!=1:
            raise ValueError('Encoder and planner TensorRT deployments differ')
    if args.check_only:
        metadata={}
        if not args.use_tensorrt:
            sd=torch.load(weight,map_location='cpu',weights_only=True)
            metadata={'score_mode':YopoNetwork.checkpoint_score_mode(sd),
                      'single_frame_no_pose':YopoNetwork.checkpoint_single_frame(sd),
                      'temporal_no_pose':YopoNetwork.checkpoint_temporal_no_pose(sd),
                      'rgb_encoder':YopoNetwork.checkpoint_rgb_encoder(sd)}
        print(json.dumps({'weight':weight,'velocity':cfg['velocity'],'vel_max_train':cfg['vel_max_train'],
            'data_timesteps':cfg['data_timesteps'],'settings':settings,'checkpoint_metadata':metadata,
            'TensorRT_deployment':deployment_metadata},ensure_ascii=False,indent=2))
    else:
        YopoNet(settings, weight)
