# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO
from .go1.go1_flat.go1_flat_config import Go1FlatCfg, Go1FlatCfgPPO
from .go1.go1_flat.go1_42flat_config import Go1WoHighCfg, Go1WoHighCfgPPO
from .go1.go1_flat.go1_45flat_config import Go1WoLVelCfg, Go1WoLVelCfgPPO
from .solo.solo9_robot import Solo9Robot
from .solo.solo9_flat_config import Solo9FlatCfg, Solo9FlatCfgPPO
from .solo8.solo8_robot import Solo8Robot
from .solo8.solo8_flat_config import Solo8FlatCfg, Solo8FlatCfgPPO
from .solo8.solo8_mmflat_config import Solo8MMFlatCfg, Solo8MMFlatCfgPPO
from .solo8.solo8_mmrough_config import Solo8MMRoughCfg, Solo8MMRoughCfgPPO


import os

from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.Multitask_registry import mm_task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )

task_registry.register( "solo8_flat", Solo8Robot, Solo8FlatCfg(), Solo8FlatCfgPPO())
mm_task_registry.register( "solo8_mmflat", Solo8Robot, Solo8MMFlatCfg(), Solo8MMFlatCfgPPO())
mm_task_registry.register( "solo8_mmrough", Solo8Robot, Solo8MMRoughCfg(), Solo8MMRoughCfgPPO())
task_registry.register( "solo9", Solo9Robot, Solo9FlatCfg(), Solo9FlatCfgPPO())
task_registry.register( "go1", LeggedRobot, Go1RoughCfg(), Go1RoughCfgPPO() )
task_registry.register( "go1_flat", LeggedRobot, Go1FlatCfg(), Go1FlatCfgPPO() )
task_registry.register( "go1_wo_lv", LeggedRobot, Go1WoLVelCfg(), Go1WoLVelCfgPPO() )
task_registry.register( "go1_wo_high", LeggedRobot, Go1WoHighCfg(), Go1WoHighCfgPPO() )