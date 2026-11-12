# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the process-wide patch isolation helper.

The helper itself is what protects the patch-heavy test modules from leaking
state into later files, so its restore contract is pinned here with synthetic
modules: the real overlays only run in the process the helper isolates.
"""

from __future__ import annotations

import sys
from types import ModuleType

from tests.conftest_patch_isolation import (
    families_of,
    isolate_globals,
    isolated,
    patch_family,
)


def _fake_module(name: str = "fake_patch") -> ModuleType:
    module = ModuleType(name)
    module.REGISTRY = ["keep"]
    module.MAPPING = {"keep": object()}
    module.FLAG = False
    return module


def test_isolate_globals_restores_only_the_named_attributes():
    module = _fake_module()
    with isolate_globals(module, "FLAG", "REGISTRY"):
        module.FLAG = True
        module.REGISTRY.append("leak")
        module.UNMANAGED = "left alone"
    assert module.FLAG is False
    assert module.REGISTRY == ["keep"]
    # Explicit API is deliberately narrow: unnamed attributes are untouched.
    assert module.UNMANAGED == "left alone"


def test_patch_family_restores_bindings_containers_and_new_names():
    module = _fake_module()
    sys.modules[module.__name__] = module
    try:
        alias = module.REGISTRY
        saved_value = module.MAPPING["keep"]
        family = patch_family(modules=(module,), sys_modules=(module.__name__,))
        family.capture()

        with isolated(family):
            module.REGISTRY.append("leak")
            assert alias == ["keep", "leak"]
            module.FLAG = True
            module.MAPPING["new"] = object()
            module._ORIGINAL = object()

        assert module.REGISTRY == ["keep"]
        # Restored in place, so aliases taken before the test see the fix too.
        assert alias == ["keep"]
        assert module.FLAG is False
        assert {"keep": saved_value} == module.MAPPING
        assert not hasattr(module, "_ORIGINAL")
    finally:
        sys.modules.pop(module.__name__, None)


def test_patch_family_removes_modules_registered_by_a_patch():
    module = _fake_module()
    registry_entry = "fake_vendored_model"
    family = patch_family(modules=(module,), sys_modules=(registry_entry,))
    family.capture()
    try:
        with isolated(family):
            sys.modules[registry_entry] = module
            module.FLAG = True
        assert registry_entry not in sys.modules
        assert module.FLAG is False
    finally:
        sys.modules.pop(registry_entry, None)


def test_restore_is_idempotent_and_discovery_is_lazy():
    module = _fake_module()
    family = patch_family(modules=(module,))
    assert family._pristine is None  # no snapshot until capture()
    family.capture()
    assert family.modules == (module,)
    with isolated(family):
        module.FLAG = True
    with isolated(family):
        module.FLAG = True
    assert module.FLAG is False


def test_families_of_normalizes_declarations():
    module = _fake_module()
    family = patch_family(modules=(module,))

    module.PATCH_ISOLATION = family
    assert families_of(module) == (family,)

    module.PATCH_ISOLATION = (family,)
    assert families_of(module) == (family,)

    module.PATCH_ISOLATION = ()
    assert families_of(module) == ()

    del module.PATCH_ISOLATION
    assert families_of(module) == ()


def test_unknown_isolated_globals_are_removed_on_exit():
    module = _fake_module()
    with isolate_globals(module, "ABSENT"):
        module.ABSENT = 1
        assert module.ABSENT == 1
    assert not hasattr(module, "ABSENT")
