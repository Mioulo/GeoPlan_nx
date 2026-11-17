"""Depth-frame planning shared by the fixed-goal and tracker ROS nodes.

No RGB inference is performed here. The caller owns inference serialization
and chooses either latest odometry or timestamp-matched tracker odometry.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from policy.poly_solver import Poly5Solver


def depth_planning_enabled(node):
    return (node.flight_enabled and node.odom_init
            and not (getattr(node, "auto_arrival_hold", False) and node.arrive))


def run_depth_expert_frame(node, depth, odom_msg):
    """Infer the supplied depth frame using one immutable planning-state snapshot."""
    if not depth_planning_enabled(node):
        return None
    with node.lock:
        pose = odom_msg.pose.pose
        velocity = odom_msg.twist.twist.linear
        start_pos = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
        start_vel = np.array([velocity.x, velocity.y, velocity.z], dtype=np.float32)
        if node.plan_from_reference:
            start_pos = np.asarray(node.desire_pos, dtype=np.float32).copy()
            start_vel = np.asarray(node.desire_vel, dtype=np.float32).copy()
        start_acc = np.asarray(node.desire_acc, dtype=np.float32).copy()
        goal = np.asarray(node.goal, dtype=np.float32).copy()
        orientation = pose.orientation
        rotation_wc = R.from_quat([
            orientation.x, orientation.y, orientation.z, orientation.w
        ]).as_matrix() @ node.Rotation_bc
    rotation_cw = rotation_wc.T
    obs = np.concatenate((rotation_cw @ start_vel, rotation_cw @ start_acc,
                          rotation_cw @ (goal - start_pos))).astype(np.float32)
    depth_input = torch.from_numpy(depth[None, None]).to(node.device, non_blocking=True)
    obs_input = torch.from_numpy(obs[None]).to(node.device, non_blocking=True)
    outputs = node.forward_depth_expert_policy(depth_input, obs_input)
    if outputs is None:
        return None
    score, endstate_w, action_id = node.select_action_from_outputs(
        outputs["endstate"][0], outputs["score"][0], rotation_wc=rotation_wc,
        safety_score=outputs['safety_score'][0] if outputs.get('safety_score') is not None else None,
        policy=getattr(node, 'depth_expert_policy', None), start_pos=start_pos,
        start_vel=start_vel, start_acc=start_acc, goal=goal,
    )
    if node.control_policy == "depth":
        end = endstate_w[action_id]
        polynomials = [Poly5Solver(start_pos[axis], start_vel[axis], start_acc[axis],
                                   start_pos[axis] + end[axis, 0], end[axis, 1],
                                   end[axis, 2], node.traj_time) for axis in range(3)]
        with node.lock:
            # A stop/arrival during inference must not install a new trajectory.
            if not depth_planning_enabled(node):
                return None
            node.optimal_poly_x, node.optimal_poly_y, node.optimal_poly_z = polynomials
            node.ctrl_time = 0.0
    return {"score": score, "endstate_w": endstate_w, "action_id": action_id,
            "snapshot": {"start_pos": start_pos, "start_vel": start_vel,
                         "start_acc": start_acc, "rotation_wc": rotation_wc}}
