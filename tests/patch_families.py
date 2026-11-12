# SPDX-License-Identifier: Apache-2.0
"""Canonical ``PATCH_ISOLATION`` declarations for the process-wide overlays.

Every ``omlx.patches.*`` overlay is installed by production code and is
process-wide on purpose: ``mlx_lm.models.laguna`` stays registered, the MTP
orchestrator keeps its sub-patches installed, and so on.  Tests only need the
isolation boundary around their own module.  Keeping the families in one place
means a test file that exercises the same overlay (or applies it indirectly,
like the MTP dispatch tests that run the pre-load hook for other model types)
shares one definition instead of growing a divergent copy.

Usage::

    from tests.patch_families import LAGUNA
    PATCH_ISOLATION = (LAGUNA,)

See ``tests/conftest_patch_isolation.py`` for the restore contract.
"""

from __future__ import annotations

import mlx_lm.models
import mlx_lm.tool_parsers

from omlx.patches import (
    bailing_hybrid,
    glm_moe_dsa,
    laguna,
    mlx_lm_mtp,
    mlx_lm_sharded_load,
    qwen35_ane_prefill,
    qwen35_oq_a8,
    qwen35_verify_qmm,
    step3p7,
)
from omlx.patches.mlx_vlm_mtp import (
    gemma4_vlm_runtime,
    glm5_next_vlm_runtime,
    inkling_vlm_runtime,
    qwen35_moe_vlm_runtime,
    qwen35_vlm_runtime,
)
from tests.conftest_patch_isolation import patch_family

BAILING = patch_family(
    modules=(bailing_hybrid, mlx_lm.models),
    sys_modules=("mlx_lm.models.bailing_hybrid",),
)

# ``maybe_apply_pre_load_patches`` falls back to an unscoped sweep that
# installs all five mlx-vlm MTP runtimes for model types it does not list
# (step3p7, glm_moe_dsa, ...). Tests of the pre-load dispatch therefore
# apply them as a side effect.
VLM_MTP_RUNTIMES = (
    qwen35_moe_vlm_runtime,
    qwen35_vlm_runtime,
    gemma4_vlm_runtime,
    inkling_vlm_runtime,
    glm5_next_vlm_runtime,
)

QWEN35_OQ_A8 = patch_family(modules=(qwen35_oq_a8,))

ANE_PREFILL = patch_family(modules=(qwen35_ane_prefill,))

LAGUNA = patch_family(
    modules=(laguna, mlx_lm.models, mlx_lm.tool_parsers),
    sys_modules=("mlx_lm.models.laguna", "mlx_lm.tool_parsers.laguna"),
)

STEP3P7 = patch_family(
    modules=(step3p7, mlx_lm.models, *VLM_MTP_RUNTIMES),
    sys_modules=("mlx_lm.models.step3p7",),
)

GLM_MOE_DSA = patch_family(
    modules=(glm_moe_dsa, mlx_lm.models, mlx_lm_sharded_load, *VLM_MTP_RUNTIMES),
    sys_modules=("mlx_lm.models.glm_moe_dsa",),
)

# The MTP orchestrator installs its own sub-patches plus the step3p7/glm_moe_dsa
# registrations that the pre-load dispatch tests exercise.
MTP = patch_family(
    modules=(mlx_lm_mtp, qwen35_verify_qmm, step3p7, glm_moe_dsa, mlx_lm.models),
    packages=("omlx.patches.mlx_lm_mtp", "omlx.patches.mlx_vlm_mtp"),
    sys_modules=("mlx_lm.models.step3p7", "mlx_lm.models.glm_moe_dsa"),
)
