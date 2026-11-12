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
import mlx_lm.models.cache
import mlx_lm.server
import mlx_lm.tool_parsers
import mlx_lm.tokenizer_utils
import mlx_lm.utils
import mlx_vlm
import mlx_vlm.models
import mlx_vlm.models.activations
import mlx_vlm.models.base
import mlx_vlm.models.mlp
import mlx_vlm.models.qwen3_5.language
import mlx_vlm.prompt_utils
import mlx_vlm.tools.parsers
import mlx_vlm.utils

from omlx import memory_monitor
from omlx.utils import model_loading
from omlx.patches import (
    arrays_cache_extract,
    bailing_hybrid,
    deepseek_v4,
    dflash_glm5,
    dflash_lifecycle,
    gemma4_audio,
    glm_moe_dsa,
    index_cache,
    k2_horizon,
    laguna,
    minimax_m3_sparse_attention,
    mlx_lm_mtp,
    mlx_lm_sharded_load,
    mlx_vlm_glm5_next_compat,
    mlx_vlm_inkling_compat,
    mlx_vlm_minimax_m3_compat,
    mlx_vlm_muse_glimmer_compat,
    mlx_vlm_unlimited_ocr_compat,
    mimo_v2,
    qwen35_ane_prefill,
    qwen35_gdn_prework,
    qwen35_moe_router,
    qwen35_oq_a8,
    qwen35_q4_mlp,
    qwen35_verify_qmm,
    qwen35_verify_sdpa_split,
    sdpa256_attention,
    step3p7,
    turboquant_attention,
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

# --- remaining 214 easy-family sites ---------------------------------------
# Each family owns only the guards and namespace entries its own overlay
# installs.  A test file that exercises several overlays declares every
# family it touches so they are restored together (see test_oq.py).

DEEPSEEK_V4 = patch_family(
    modules=(
        deepseek_v4,
        mlx_lm.models,
        mlx_lm.models.cache,
        mlx_lm.server,
        mlx_lm.utils,
        mlx_lm.tokenizer_utils,
        model_loading,
    ),
    packages=("omlx.patches.deepseek_v4",),
    sys_modules=(
        "mlx_lm.models.deepseek_v4",
        "mlx_lm.models.deepseek_v4_mtp",
        "mlx_lm.models.hyper_connection",
        "mlx_lm.chat_templates.deepseek_v4",
        "mlx_lm.tool_parsers.deepseek_v4",
    ),
)

# The vendor tree only exists after the overlay extends mlx_vlm.__path__, so the
# factory registers the exact module names its vendor files provide.
GLM5_NEXT = patch_family(
    modules=(
        mlx_vlm_glm5_next_compat,
        glm5_next_vlm_runtime,
        deepseek_v4,
        mlx_lm.models,
        mlx_lm.models.cache,
        mlx_vlm,
        mlx_vlm.models,
        mlx_vlm.prompt_utils,
        mlx_vlm.utils,
    ),
    packages=("omlx.patches.mlx_vlm_glm5_next_compat",),
    sys_modules=(
        "mlx_vlm.models.glm5_next",
        "mlx_vlm.models.glm5_next.config",
        "mlx_vlm.models.glm5_next.vision",
        "mlx_vlm.models.glm5_next.linear",
        "mlx_vlm.models.glm5_next.glm5_next",
        "mlx_vlm.models.glm5_next.language",
        "mlx_vlm.models.glm5_next.hc_prefill",
        "mlx_vlm.models.glm5_next.processing",
        # Applying this overlay also installs the pooling cache. Some tests
        # (oq round-trips) go on to load a deepseek_v4 model, which registers
        # these too; put them back with the same boundary.
        "mlx_lm.models.deepseek_v4",
        "mlx_lm.models.deepseek_v4_mtp",
        "mlx_lm.models.hyper_connection",
        "mlx_lm.chat_templates.deepseek_v4",
        "mlx_lm.tool_parsers.deepseek_v4",
    ),
)

MINIMAX_M3 = patch_family(
    modules=(
        mlx_vlm_minimax_m3_compat,
        mlx_vlm,
        mlx_vlm.models,
        mlx_vlm.utils,
        mlx_vlm.prompt_utils,
        mlx_vlm.tools.parsers,
    ),
    sys_modules=(
        "mlx_vlm.models.minimax_m3",
        "mlx_vlm.models.minimax_m3.minimax_m3",
        "mlx_vlm.models.minimax_m3_vl",
        "mlx_vlm.models.minimax_m3_vl.config",
        "mlx_vlm.models.minimax_m3_vl.language",
        "mlx_vlm.models.minimax_m3_vl.minimax_m3_vl",
        "mlx_vlm.models.minimax_m3_vl.msa",
        "mlx_vlm.models.minimax_m3_vl.processing_minimax_m3_vl",
        "mlx_vlm.models.minimax_m3_vl.vision",
        "mlx_vlm.tools.parsers.minimax_m3",
    ),
)

# Inkling's vendor tree also shadows the shared activations/mlp shims, so those
# two modules are part of its family.
INKLING = patch_family(
    modules=(
        mlx_vlm_inkling_compat,
        inkling_vlm_runtime,
        mlx_vlm,
        mlx_vlm.models,
        mlx_vlm.models.activations,
        mlx_vlm.models.mlp,
        mlx_vlm.utils,
        mlx_vlm.prompt_utils,
    ),
    sys_modules=(
        "mlx_vlm.models.inkling",
        "mlx_vlm.models.inkling.config",
        "mlx_vlm.models.inkling.vision",
        "mlx_vlm.models.inkling.inkling",
        "mlx_vlm.models.inkling.processing_inkling",
        "mlx_vlm.models.inkling.language",
        "mlx_vlm.models.inkling.audio",
        "mlx_vlm.models.inkling.switch_layers",
    ),
)

UNLIMITED_OCR = patch_family(
    modules=(
        mlx_vlm_unlimited_ocr_compat,
        mlx_vlm,
        mlx_vlm.models,
        mlx_vlm.utils,
        mlx_vlm.prompt_utils,
    ),
    sys_modules=(
        "mlx_vlm.models.unlimited_ocr",
        "mlx_vlm.models.unlimited_ocr.config",
        "mlx_vlm.models.unlimited_ocr.processing_unlimitedocr",
        "mlx_vlm.models.unlimited_ocr.language",
        "mlx_vlm.models.unlimited_ocr.unlimitedocr",
    ),
)

MUSE_GLIMMER = patch_family(
    modules=(
        mlx_vlm_muse_glimmer_compat,
        mlx_vlm,
        mlx_vlm.models,
        mlx_vlm.models.activations,
        mlx_vlm.prompt_utils,
    ),
    sys_modules=(
        "mlx_vlm.models.muse_glimmer",
        "mlx_vlm.models.muse_glimmer.config",
        "mlx_vlm.models.muse_glimmer.vision",
        "mlx_vlm.models.muse_glimmer.language",
        "mlx_vlm.models.muse_glimmer.muse_glimmer",
        "mlx_vlm.models.muse_glimmer.processing_muse_glimmer",
    ),
)

# TurboQuant/sdpa256 rebind scaled_dot_product_attention on every already
# imported model module, so their family snapshots everything under those
# prefixes at collection time (module_prefixes never imports anything).
TURBOQUANT = patch_family(
    modules=(turboquant_attention, mlx_lm.models, mlx_lm.models.base, mlx_vlm.models.base),
    module_prefixes=("mlx_lm.models.", "mlx_vlm.models."),
)

SDPA256 = patch_family(
    modules=(sdpa256_attention, memory_monitor, mlx_lm.models, mlx_lm.models.base),
    module_prefixes=("mlx_lm.models.", "mlx_vlm.models."),
)

DFLASH = patch_family(
    modules=(dflash_lifecycle, dflash_glm5),
)

INDEX_CACHE = patch_family(
    modules=(index_cache, mlx_lm.models, mlx_vlm.models),
)

GDN_PREWORK = patch_family(
    modules=(qwen35_gdn_prework, mlx_vlm.models, mlx_vlm.models.qwen3_5.language),
)

Q4_MLP = patch_family(
    modules=(qwen35_q4_mlp, mlx_vlm.models, mlx_vlm.models.qwen3_5.language),
)

M3_SPARSE = patch_family(
    modules=(minimax_m3_sparse_attention, mlx_lm.models),
)

MOE_ROUTER = patch_family(
    modules=(qwen35_moe_router, mlx_vlm.models, mlx_vlm.models.qwen3_5.language),
)

VERIFY_SPLIT = patch_family(
    modules=(
        qwen35_verify_sdpa_split,
        mlx_vlm.models,
        mlx_vlm.models.qwen3_5.language,
    ),
)

GEMMA4 = patch_family(
    modules=(gemma4_audio, gemma4_vlm_runtime, mlx_lm.models),
)

ARRAYS = patch_family(
    modules=(arrays_cache_extract, mlx_lm.models.cache, mlx_vlm.models.cache),
)

K2_HORIZON = patch_family(
    modules=(k2_horizon, mlx_lm.models, mlx_lm.tool_parsers),
    packages=("omlx.patches.k2_horizon",),
    sys_modules=("mlx_lm.models.k2_horizon", "mlx_lm.tool_parsers.k2_horizon"),
)

MIMO_V2 = patch_family(
    modules=(mimo_v2, mlx_lm.models),
    packages=("omlx.patches.mimo_v2",),
    sys_modules=("mlx_lm.models.mimo_v2", "mlx_lm.models.mimo_v2_flash"),
)
