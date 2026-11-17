"""Bind TensorRT artifacts to their score semantics and exported configuration."""
import hashlib
import json
import math
from pathlib import Path


SCHEMA = 'yopo.temporal-engine.v1'
RUNTIME_KEYS = ('image_height', 'image_width', 'data_timesteps', 'vertical_num', 'horizon_num',
                'velocity', 'vel_max_train', 'acc_max_train', 'sgm_time', 'radio_range',
                'horizon_camera_fov', 'vertical_camera_fov', 'horizon_anchor_fov', 'vertical_anchor_fov')


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def planning_output_names(mode):
    if mode not in ('selection', 'collision', 'topk-collision'):
        raise ValueError('Invalid engine score mode')
    return ['endstate', 'score'] + (['safety_score'] if mode == 'topk-collision' else [])


def make_metadata(policy, role, config):
    source = getattr(policy, 'deployment_weight_sha256', None)
    if not source:
        raise ValueError('Export metadata requires a checkpoint loaded through load_policy')
    info = dict(schema=SCHEMA, role=role, score_mode=policy.score_mode,
                candidate_topk=policy.candidate_topk, weight_sha256=source,
                runtime_config={key: config[key] for key in RUNTIME_KEYS},
                input_names=['rgb'] if role == 'encoder' else ['features', 'obs', 'pose'] if role == 'planner' else ['rgb', 'obs', 'pose'],
                output_names=['features'] if role == 'encoder' else planning_output_names(policy.score_mode))
    identity = {key: value for key, value in info.items() if key not in ('role', 'input_names', 'output_names')}
    info['deployment_id'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return info


def write_metadata(path, metadata, artifact):
    info = dict(metadata)
    info[artifact + '_sha256'] = file_sha256(path)
    destination = Path(str(path) + '.json')
    temporary = destination.with_name(destination.name + '.writing')
    temporary.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(destination)


def read_metadata(path, artifact='engine'):
    sidecar = Path(str(path) + '.json')
    if not sidecar.exists():
        return None  # Existing classification-only engines remain supported.
    info = json.loads(sidecar.read_text(encoding='utf-8'))
    if info.get('schema') != SCHEMA or info.get('role') not in ('full', 'encoder', 'planner'):
        raise ValueError(f'Invalid engine metadata: {sidecar}')
    names = ['features'] if info['role'] == 'encoder' else planning_output_names(info.get('score_mode'))
    if info.get('output_names') != names:
        raise ValueError('Engine output names disagree with score semantics')
    count = int(info['runtime_config']['vertical_num']) * int(info['runtime_config']['horizon_num'])
    if not 1 <= int(info.get('candidate_topk', 0)) <= count:
        raise ValueError('Invalid engine candidate_topk')
    if info.get(artifact + '_sha256') != file_sha256(path):
        raise ValueError(f'{path} differs from its .json; copy matching artifacts')
    return info


def validate_runtime_config(info, config):
    if info is None:
        return
    for key in RUNTIME_KEYS:
        if not math.isclose(float(info['runtime_config'][key]), float(config[key]), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f'Engine {key}={info["runtime_config"][key]} differs from current {config[key]}; rebuild or restore export configuration')
