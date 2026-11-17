#!/usr/bin/env python3
"""ROS1 deployment entry for the 116 split TensorRT engines."""
import argparse
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = HERE if (HERE / 'policy').is_dir() else HERE.parents[1] / 'YOPO'
sys.path.insert(0, str(RUNTIME))

import numpy as np
import rospy
import torch
from ruamel.yaml import YAML
from config.config import cfg
from policy.engine_metadata import read_metadata, validate_runtime_config
from test_yopo_ros import YopoNet


def resolve_path(value, base):
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def load_settings(config_path):
    config_path = Path(config_path).resolve()
    options = YAML(typ='safe').load(config_path.read_text())
    if not isinstance(options, dict):
        raise ValueError('Deployment YAML must contain a mapping')
    base = config_path.parent
    training = resolve_path(options.get('model_config', 'saved/YOPO_116/ablation_config.json'), base)
    with training.open() as stream:
        cfg._data.update(json.load(stream))
    cfg['train'] = False
    if 'velocity' in options:
        cfg['velocity'] = float(options['velocity'])
    if not math.isfinite(cfg['velocity']) or cfg['velocity'] <= 0:
        raise ValueError('velocity must be finite and positive')
    cfg['goal_length'] = 2 * cfg['radio_range']
    cfg['sgm_time'] = 2 * cfg['radio_range'] / cfg['vel_max_train']
    cfg['traj_num'] = cfg['horizon_num'] * cfg['vertical_num'] * cfg['radio_num']
    engines, metadata = {}, {}
    for role in ('encoder', 'planner'):
        engines[role] = str(resolve_path(options.get(role + '_engine', 'proposed116_' + role + '.engine'), base))
        info = read_metadata(engines[role])
        if info is None:
            raise ValueError('Missing engine metadata: ' + engines[role] + '.json')
        validate_runtime_config(info, cfg)
        if info['role'] != role or info['score_mode'] != 'topk-collision' or info['candidate_topk'] != 8:
            raise ValueError('Expected 116 Top8 dual-score ' + role + ' engine')
        metadata[role] = info
    if metadata['encoder']['deployment_id'] != metadata['planner']['deployment_id']:
        raise ValueError('Encoder and planner must come from the same export')
    expected_sha = resolve_path(options.get('checkpoint_sha_file', 'checkpoint.sha256'), base).read_text().strip()
    if any(info['weight_sha256'] != expected_sha for info in metadata.values()):
        raise ValueError('Engine checkpoint hash differs from the packaged 116 checkpoint')
    goal = options.get('initial_goal')
    if goal is not None and (len(goal) != 3 or not np.isfinite(goal).all()):
        raise ValueError('initial_goal must be null or three finite coordinates')
    goal_height = options.get('goal_height')
    if goal_height is not None and not math.isfinite(float(goal_height)):
        raise ValueError('goal_height must be null or finite')
    tolerance = float(options.get('arrival_tolerance', 1.0))
    pitch = float(options.get('pitch_angle_deg', 0.0))
    if not math.isfinite(tolerance) or tolerance <= 0 or not math.isfinite(pitch):
        raise ValueError('Invalid arrival_tolerance or pitch_angle_deg')
    topics = {key: options.get(key, default) for key, default in (
        ('rgb_topic', '/rgb/image'), ('odom_topic', '/sim/odom'),
        ('ctrl_topic', '/so3_control/pos_cmd'), ('goal_topic', '/move_base_simple/goal'))}
    if any(not isinstance(value, str) or not value.strip() for value in topics.values()):
        raise ValueError('Topic names must be nonempty strings')
    settings = dict(
        use_tensorrt=True, use_feature_cache=True,
        encoder_engine=engines['encoder'], planner_engine=engines['planner'],
        goal=[0., 0., 0.] if goal is None else goal,
        goal_height=goal_height, pitch_angle_deg=pitch,
        plan_from_reference=bool(options.get('plan_from_reference', False)),
        control_policy='rgb', selection_strategy='top8-cost',
        publish_r2d_depth=False, publish_feature_visualization=False,
        publish_depth_expert_traj=False, validation_mode=False,
        auto_arrival_hold=bool(options.get('auto_arrival_hold', True)),
        arrival_tolerance=tolerance, visualize=False,
        verbose=bool(options.get('verbose', False)), **topics,
    )
    return settings, goal is not None


class DeploymentNode(YopoNet):
    """Reuse established planning/control methods with configurable TRT inputs."""
    def __init__(self, settings, has_goal):
        self.has_goal = has_goal
        self.arrival_hold_requested = settings['auto_arrival_hold']
        settings = dict(settings, auto_arrival_hold=has_goal and self.arrival_hold_requested)
        sys.argv.append('/move_base_simple/goal:=' + settings['goal_topic'])
        super().__init__(settings, settings['planner_engine'])

    def callback_set_goal(self, data):
        if not np.isfinite([data.pose.position.x, data.pose.position.y, data.pose.position.z]).all():
            rospy.logwarn('Ignoring nonfinite goal')
            return
        with self.rgb_lock:
            super().callback_set_goal(data)
            self.has_goal = True
            self.auto_arrival_hold = self.arrival_hold_requested

    def _callback_rgb_locked(self, data):
        if not self.has_goal:
            rospy.loginfo_throttle(5., '116 TRT ready; waiting for a goal')
            return
        return super()._callback_rgb_locked(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'deployment.yaml')
    parser.add_argument('--check_only', action='store_true', help='validate artifacts/configuration without ROS startup')
    args, remaining = parser.parse_known_args()
    if any(':=' not in arg for arg in remaining):
        parser.error('Unknown arguments: ' + ' '.join(remaining))
    settings, has_goal = load_settings(args.config)
    if args.check_only:
        print(json.dumps(dict(settings=settings, velocity=cfg['velocity'], initial_goal_set=has_goal), indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('116 TRT deployment requires CUDA-enabled Jetson PyTorch; CPU fallback is disabled')
    DeploymentNode(settings, has_goal)


if __name__ == '__main__':
    main()
