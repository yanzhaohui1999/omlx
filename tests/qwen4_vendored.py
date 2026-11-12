# SPDX-License-Identifier: Apache-2.0
"""Shared isolation plumbing for tests of oMLX's vendored Qwen4-Exp model.

``mlx_vlm.models.qwen4_exp`` only exists once
``apply_mlx_vlm_qwen4_exp_compat_patch()`` has extended ``mlx_vlm.__path__``, so
test modules used to call the patch at import time.  Import-time patching
leaks into every later test in the pytest worker; install it from the
module-scoped ``PATCH_ISOLATION_SETUP`` fixture instead (see
``tests/conftest_patch_isolation.py``) and let conftest restore the state.

``QWEN4_COMPAT`` snapshots only the overlay's own ``_APPLIED`` guard. The
shared ``mlx_vlm`` namespaces are left to the module that owns each wrapper
(inkling/minimax compat keep their guards elsewhere), so restoring this family
cannot strip a sibling overlay's hooks.
"""

from __future__ import annotations

import importlib
from collections.abc import MutableMapping
from typing import Any

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from tests.conftest_patch_isolation import patch_family

# Only the overlay's own guard is snapshotted. The vendor search paths and the
# ``mlx_vlm.utils`` / ``prompt_utils`` wrappers are deliberately *not* restored:
# the inkling/minimax compat overlays patch the same shared namespaces and keep
# their ``_APPLIED`` guards in their own modules, so restoring ours would strip
# their hooks without letting them re-install.
QWEN4_COMPAT = patch_family(modules=(compat,))


def apply_compat() -> None:
    """Expose the vendored implementation (idempotent)."""
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()


def import_into(namespace: MutableMapping[str, Any], *paths: str) -> Any:
    """Apply the compat overlay and bind each imported module into ``namespace``.

    ``import_into(globals(), "mlx_vlm.models.qwen4_exp.language")`` binds
    ``language`` so the rest of the test module can use it as before, while
    the import itself only happens inside the isolated fixture window.  Pass
    ``"dotted.path=alias"`` when the module name alone is not a unique local
    name (``qwen3_5.language`` vs ``qwen4_exp.language``).
    """
    apply_compat()
    imported = []
    for spec in paths:
        path, _, alias = spec.partition("=")
        module = importlib.import_module(path)
        namespace[alias or path.rsplit(".", 1)[-1]] = module
        imported.append(module)
    return imported[0] if len(imported) == 1 else imported
