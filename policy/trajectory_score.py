"""Analytic jerk/acceleration/goal costs plus the learned weighted collision cost.

Restored from commit 075bb3b.
"""
import math
import numpy as np
from config.config import cfg


def choose_candidate(total_cost, selection_logits=None, candidate_topk=8):
    """Minimize total cost within the highest-logit candidates, or all candidates."""
    costs = np.asarray(total_cost).reshape(-1)
    if selection_logits is None:
        return int(np.argmin(costs))
    logits = np.asarray(selection_logits).reshape(-1)
    if logits.shape != costs.shape:
        raise ValueError('Logits and costs must have the same candidate count')
    k = min(max(1, int(candidate_topk)), len(costs))
    candidates = np.argpartition(-logits, k-1)[:k]
    return int(candidates[np.argmin(costs[candidates])])


class ClosedFormTrajectoryScore:
    def __init__(self, traj_time):
        self.traj_time = traj_time
        self.RJ, self.RA = self._qp_generation(traj_time)
        vel_scale = 2.0 * cfg["radio_range"] / traj_time
        self.smoothness_weight = cfg["ws"] / vel_scale ** 5
        self.accel_weight = cfg["wa"] / vel_scale ** 3
        self.goal_weight = cfg["wg"]

    def _qp_generation(self, traj_time):
        A = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            A[2 * i, i] = math.factorial(i)
            for j in range(i, 6):
                A[2 * i + 1, j] = math.factorial(j) / math.factorial(j - i) * (traj_time ** (j - i))

        H = np.zeros((6, 6), dtype=np.float64)
        for i in range(3, 6):
            for j in range(3, 6):
                H[i, j] = i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5) * (traj_time ** (i + j - 5))

        Q = np.zeros((6, 6), dtype=np.float64)
        for i in range(2, 6):
            for j in range(2, 6):
                Q[i, j] = (i * (i - 1)) * (j * (j - 1)) / (i + j - 3) * (traj_time ** (i + j - 3))

        Ct = np.zeros((6, 6), dtype=np.float64)
        Ct[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1.0
        C = Ct.T
        B = np.linalg.inv(A)
        return C @ B.T @ H @ B @ Ct, C @ B.T @ Q @ B @ Ct

    def total_score(self, safety_score, rel_endstate_w, start_pos, start_vel, start_acc, goal):
        traj_num = rel_endstate_w.shape[0]
        start_pos = np.asarray(start_pos, dtype=np.float64)
        start_vel = np.asarray(start_vel, dtype=np.float64)
        start_acc = np.asarray(start_acc, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        Df = np.stack([
            np.repeat(start_pos[None, :], traj_num, axis=0),
            np.repeat(start_vel[None, :], traj_num, axis=0),
            np.repeat(start_acc[None, :], traj_num, axis=0),
        ], axis=2)
        Dp = np.stack([
            rel_endstate_w[:, :, 0] + start_pos[None, :],
            rel_endstate_w[:, :, 1],
            rel_endstate_w[:, :, 2],
        ], axis=2)
        D_all = np.concatenate([Df, Dp], axis=2)

        smooth = self._quadratic_cost(D_all, self.RJ)
        accel = self._quadratic_cost(D_all, self.RA)
        guidance = self._guidance_cost(Df[:, :, 0], Dp[:, :, 0], goal)
        return (
            np.asarray(safety_score, dtype=np.float64)
            + self.smoothness_weight * smooth
            + self.accel_weight * accel
            + self.goal_weight * guidance
        )

    def _quadratic_cost(self, D_all, R_cost):
        return sum(
            np.einsum("ni,ij,nj->n", D_all[:, axis, :], R_cost, D_all[:, axis, :])
            for axis in range(3)
        )

    def _guidance_cost(self, cur_pos, end_pos, goal):
        traj_dir = end_pos - cur_pos
        goal_dir = goal[None, :] - cur_pos
        goal_norm = np.linalg.norm(goal_dir, axis=1, keepdims=True)
        goal_dir_norm = goal_dir / np.maximum(goal_norm, 1e-8)
        traj_along = np.sum(traj_dir * goal_dir_norm, axis=1)
        goal_length = goal_norm[:, 0]
        abs_diff = np.abs(goal_length - traj_along)
        parallel_diff = np.where(abs_diff < 1.0, 0.5 * abs_diff ** 2, abs_diff - 0.5)
        traj_perp = traj_dir - traj_along[:, None] * goal_dir_norm
        return parallel_diff + 0.5 * np.linalg.norm(traj_perp, axis=1)
