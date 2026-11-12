# SPDX-License-Identifier: Apache-2.0
"""Save/restore helpers for tests that install process-wide monkey-patches.

Several ``omlx.patches.*`` overlays are process-wide by design: ``apply_*()``
rebinds classes in ``mlx-lm`` / ``mlx-vlm``, registers vendored modules in
``sys.modules`` and flips ``_APPLIED`` guards.  A test that calls one of them
without a teardown changes how every later test in the same pytest worker
behaves, which makes failures depend on file order instead of on the code
under test (see the #5176 post-mortem).  pytest-xdist hides this locally
because each worker owns a different slice of files.

This module records a *pristine* snapshot of the patch modules and their
victims right after collection (before any test runs) and restores it around
each test module that declares an interest in it:

    # in a test module
    from tests.conftest_patch_isolation import patch_family
    from omlx.patches import laguna, step3p7

    PATCH_ISOLATION = (
        patch_family(modules=(laguna,), sys_modules=("mlx_lm.models.laguna",)),
        patch_family(modules=(step3p7,), sys_modules=("mlx_lm.models.step3p7",)),
    )

``tests/conftest.py`` installs an autouse module-scoped fixture that restores
the pristine snapshot *before* the module's first test (healing leakage from an
earlier module) and again *after* its last one (stopping its own leakage).

Discovery is automatic within the modules a test file names: every module-level
attribute that is a mutable container is copied and restored in place, every
other attribute is compared by identity and put back if it was rebound.  That
covers ``_APPLIED`` flags, ``_GDN_MODULES``-style registries and package
``__path__`` lists without listing a single global by name.  ``packages=``
extends the snapshot to every submodule of a patch package, which is what the
orchestrator-style patches need.

Limit: only *module-level* state is restored.  Class-level monkey-patches
(``TextModel.__call__ = ...``) are production behavior and stay installed;
tests that assert stock behavior must unwind those themselves (as they already
do with ``monkeypatch``).  Objects nested inside a mutable container are
compared by reference only; mutate a nested dict in place and this helper will
not see it.

A one-off, explicit alternative is available when a test only touches a few
names:

    with isolate_globals(memory_monitor, "_SDPA_TILED_PREFILL_HEAD_DIMS"):
        ...
"""

from __future__ import annotations

import copy
import importlib
import pkgutil
import sys
from collections.abc import Iterable, Iterator, MutableMapping, Sequence
from contextlib import contextmanager
from types import ModuleType

__all__ = [
    "PatchFamily",
    "capture_pristine",
    "families_from",
    "families_of",
    "isolate_globals",
    "isolated",
    "patch_family",
]

_MISSING = object()

# Module attributes pytest/importlib own; never restored or deleted.
_SKIP_ATTRS = frozenset(
    {"__builtins__", "__loader__", "__spec__", "__cached__"}
)

_CONTAINERS = (list, set, bytearray, MutableMapping)


def _is_container(value: object) -> bool:
    return isinstance(value, _CONTAINERS)


def _copy_container(value: object) -> object:
    if isinstance(value, (list, set, MutableMapping)):
        return copy.copy(value)
    return bytearray(value)  # type: ignore[arg-type]


def _restore_container(target: object, saved: object) -> None:
    """Restore ``target`` in place so aliases held elsewhere see the fix."""
    if isinstance(target, list):
        target[:] = saved  # type: ignore[index]
    elif isinstance(target, (MutableMapping, set)):
        target.clear()
        target.update(saved)  # type: ignore[arg-type]
    elif isinstance(target, bytearray):
        target[:] = saved  # type: ignore[index]


def _same_container(current: object, saved: object) -> bool:
    """Compare shallow container contents without invoking array ``==``."""
    if type(current) is not type(saved) or len(current) != len(saved):  # type: ignore[arg-type]
        return False
    if isinstance(current, list):
        return all(a is b for a, b in zip(current, saved))  # type: ignore[arg-type]
    if isinstance(current, dict):
        return current.keys() == saved.keys() and all(  # type: ignore[union-attr]
            current[key] is saved[key] for key in current  # type: ignore[index]
        )
    return current == saved


class _ModuleSnapshot:
    """Pristine copy of one module's namespace."""

    def __init__(self, module: ModuleType) -> None:
        self.module = module
        self.attrs: dict[str, object] = {}
        self.mutables: dict[str, object] = {}
        for name, value in vars(module).items():
            if name in _SKIP_ATTRS:
                continue
            self.attrs[name] = value
            if _is_container(value):
                self.mutables[name] = _copy_container(value)

    def leaks(self) -> list[str]:
        """Names that still differ from the pristine snapshot."""
        namespace = vars(self.module)
        changed = [
            name
            for name, original in self.attrs.items()
            if namespace.get(name, _MISSING) is not original
        ]
        for name, saved in self.mutables.items():
            if name in changed:
                continue
            if not _same_container(namespace.get(name, _MISSING), saved):
                changed.append(name)
        changed.extend(
            name
            for name in namespace
            if name not in self.attrs and name not in _SKIP_ATTRS
        )
        return changed

    def restore(self) -> None:
        namespace = vars(self.module)

        # Rebindings and in-place container mutation.
        for name, original in self.attrs.items():
            saved = self.mutables.get(name, _MISSING)
            current = namespace.get(name, _MISSING)
            if saved is not _MISSING:
                # The pristine object may itself have been mutated before it
                # was replaced; restore its contents either way.
                _restore_container(original, saved)
                if current is not original:
                    namespace[name] = original
                continue
            if current is not original:
                namespace[name] = original

        # Names the patch added after the snapshot (``_ORIGINAL_*``,
        # ``_APPLIED`` on a lazily imported module, ...).
        for name in list(namespace):
            if name not in self.attrs and name not in _SKIP_ATTRS:
                del namespace[name]


class PatchFamily:
    """A set of modules (and sys.modules entries) patched as one unit."""

    def __init__(
        self,
        *,
        modules: Iterable[ModuleType] = (),
        packages: Iterable[str] = (),
        sys_modules: Iterable[str] = (),
    ) -> None:
        self._modules = tuple(modules)
        self._packages = tuple(packages)
        self._sys_modules = tuple(sys_modules)
        self._pristine: tuple[tuple[_ModuleSnapshot, ...], dict[str, ModuleType]] | None = None

    # -- discovery ---------------------------------------------------------

    def _package_members(self) -> list[ModuleType]:
        members: list[ModuleType] = []
        for package_name in self._packages:
            package = importlib.import_module(package_name)
            members.append(package)
            path = getattr(package, "__path__", None)
            if path is None:
                continue
            for info in pkgutil.walk_packages(path, package.__name__ + "."):
                try:
                    members.append(importlib.import_module(info.name))
                except Exception:  # noqa: BLE001 - optional backends
                    continue
        return members

    def _members(self) -> list[ModuleType]:
        seen: set[int] = set()
        members: list[ModuleType] = []
        for module in (*self._modules, *self._package_members()):
            if id(module) in seen:
                continue
            seen.add(id(module))
            members.append(module)
        return members

    # -- snapshot / restore -------------------------------------------------

    def capture(self) -> None:
        """Record the current state as pristine. Idempotent."""
        if self._pristine is not None:
            return
        snapshots = tuple(_ModuleSnapshot(module) for module in self._members())
        existing = {
            name: sys.modules[name]
            for name in self._sys_modules
            if name in sys.modules
        }
        self._pristine = (snapshots, existing)

    def restore(self) -> None:
        """Put every captured module and sys.modules entry back."""
        if self._pristine is None:
            self.capture()
        assert self._pristine is not None
        snapshots, existing = self._pristine
        for snapshot in snapshots:
            snapshot.restore()
        for name in self._sys_modules:
            if name in existing:
                sys.modules[name] = existing[name]
            else:
                sys.modules.pop(name, None)

    def leaks(self) -> dict[str, list[str]]:
        """Modules in this family whose state no longer matches the snapshot."""
        if self._pristine is None:
            return {}
        assert self._pristine is not None
        snapshots, existing = self._pristine
        leaked: dict[str, list[str]] = {}
        for snapshot in snapshots:
            names = snapshot.leaks()
            if names:
                leaked[snapshot.module.__name__] = names
        missing = [name for name in existing if name not in sys.modules]
        if missing:
            leaked["sys.modules"] = missing
        return leaked

    # -- introspection ------------------------------------------------------

    @property
    def modules(self) -> tuple[ModuleType, ...]:
        if self._pristine is None:
            return tuple(self._members())
        assert self._pristine is not None
        return tuple(snapshot.module for snapshot in self._pristine[0])


def patch_family(
    *,
    modules: Iterable[ModuleType] = (),
    packages: Iterable[str] = (),
    sys_modules: Iterable[str] = (),
) -> PatchFamily:
    """Build a :class:`PatchFamily`; see the module docstring for usage."""
    return PatchFamily(modules=modules, packages=packages, sys_modules=sys_modules)


def capture_pristine(families: Iterable[PatchFamily]) -> None:
    """Freeze the pre-test state of ``families`` (called after collection)."""
    seen: set[int] = set()
    for family in families:
        if id(family) in seen:
            continue
        seen.add(id(family))
        family.capture()


@contextmanager
def isolated(*families: PatchFamily) -> Iterator[None]:
    """Restore pristine state before and after the wrapped block."""
    for family in families:
        family.capture()
        family.restore()
    try:
        yield
    finally:
        for family in reversed(families):
            family.restore()


@contextmanager
def isolate_globals(module: ModuleType, *names: str) -> Iterator[None]:
    """Explicit variant: restore only the named module attributes."""
    saved: dict[str, tuple[object, object]] = {}
    for name in names:
        if not hasattr(module, name):
            saved[name] = (_MISSING, _MISSING)
            continue
        value = getattr(module, name)
        saved[name] = (
            value,
            _copy_container(value) if _is_container(value) else _MISSING,
        )
    try:
        yield
    finally:
        for name, (original, contents) in saved.items():
            if original is _MISSING:
                if hasattr(module, name):
                    delattr(module, name)
                continue
            if contents is not _MISSING:
                _restore_container(original, contents)
            setattr(module, name, original)


def families_of(module: object) -> tuple[PatchFamily, ...]:
    """Normalize a module's ``PATCH_ISOLATION`` declaration to a tuple."""
    declared = getattr(module, "PATCH_ISOLATION", ()) or ()
    if isinstance(declared, PatchFamily):
        return (declared,)
    return tuple(declared)


def families_from(items: Sequence[object]) -> list[PatchFamily]:
    """Collect ``PATCH_ISOLATION`` declarations from collected test items."""
    families: list[PatchFamily] = []
    seen: set[int] = set()
    for item in items:
        try:
            module = item.module  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - doctest/plugin items
            continue
        for family in families_of(module):
            if id(family) in seen:
                continue
            seen.add(id(family))
            families.append(family)
    return families
