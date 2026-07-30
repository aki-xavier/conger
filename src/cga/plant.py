"""Conversions between physics plant state and CGA objects.

Plant-agnostic: works with any plant exposing `body_pose(name)` in the
project convention (position + `(w, x, y, z)` quaternion), e.g.
`simu.physics.drake_plant.DrakePlant`.
"""

import math
from typing import Any

import numpy as np

from cga.algebra import gp, reverse
from cga.motors import motor_to_matrix, rotor_from_quaternion, translator
from cga.multivector import Multivector


def pose_to_motor(
    position: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
) -> Multivector:
    """Convert a world pose to a CGA motor.

    Args:
        position: World position `(x, y, z)`.
        quaternion: World quaternion `(w, x, y, z)`.
    """
    return gp(translator(position), rotor_from_quaternion(quaternion))


def plant_body_motor(plant: Any, body_name: str) -> Multivector:
    """Read one body pose from a plant and convert it to a CGA motor."""
    pose = plant.body_pose(body_name)
    return pose_to_motor(pose["position"], pose["quaternion"])


def motor_pose_error(current: Multivector, target: Multivector) -> Multivector:
    """Return the relative motor taking `current` to `target`.

    This is a compact CGA-space pose error primitive; specific controllers can
    project/log this motor into task-space or joint-space commands.
    """
    return gp(reverse(current), target)


def motor_position(M: Multivector) -> tuple[float, float, float]:
    """Extract translation from a motor via its homogeneous matrix.

    TODO: read e1∞/e2∞/e3∞ components directly from M.values for a
    lighter path that avoids constructing the full 4×4 matrix.
    """
    matrix = motor_to_matrix(M)
    return (float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3]))


def matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to `(w, x, y, z)` quaternion."""
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return (
        (m[1, 0] - m[0, 1]) / s,
        (m[0, 2] + m[2, 0]) / s,
        (m[1, 2] + m[2, 1]) / s,
        0.25 * s,
    )
