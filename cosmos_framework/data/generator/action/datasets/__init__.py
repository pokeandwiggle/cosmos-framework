# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action dataset wrappers for Cosmos Action.

All concrete datasets inherit from :class:`ActionBaseDataset` and expose a
``load_action_stats()`` classmethod for retrieving pre-computed normalization
statistics without instantiating the dataset.

The datasets are imported on first access, so importing one module of this
package (e.g. :mod:`.action_sft_wrappers`) does not require the LeRobot and
pandas dependencies of the concrete datasets.
"""

import importlib
from typing import Any

_MODULE_BY_NAME = {
    "ActionBaseDataset": "base_dataset",
    "AgiBotWorldBetaLeRobotDataset": "agibotworld_beta_lerobot_dataset",
    "BridgeOrigLeRobotDataset": "bridge_orig_lerobot_dataset",
    "DROIDLeRobotDataset": "droid_lerobot_dataset",
    "DROIDMergedLeRobotDataset": "droid_merged_lerobot_dataset",
    "FractalLeRobotDataset": "fractal_lerobot_dataset",
    "HumanHandPoseLeRobotDataset": "human_hand_pose_lerobot_dataset",
    "LIBEROLeRobotDataset": "libero_lerobot_dataset",
    "RoboCasaLeRobotDataset": "robocasa_lerobot_dataset",
    "RoboMINDFrankaDataset": "robomind_franka_dataset",
    "RoboMINDURDataset": "robomind_ur_dataset",
    "UMILeRobotDataset": "umi_lerobot_dataset",
}

__all__ = sorted(_MODULE_BY_NAME)


def __getattr__(name: str) -> Any:
    if name not in _MODULE_BY_NAME:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{_MODULE_BY_NAME[name]}"), name)
