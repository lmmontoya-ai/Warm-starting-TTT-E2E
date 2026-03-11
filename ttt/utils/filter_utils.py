"""Path-based parameter filtering helpers.

These utilities mirror the reference filtering semantics closely enough for the
warm-start/runtime parity work while staying implemented in the main repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
from jax._src.lib import pytree
from jaxtyping import PyTree


@dataclass(frozen=True)
class SpecNode:
    @staticmethod
    def from_string(string: str) -> "SpecNode":
        if string == "**":
            return DoubleWildNode()
        if string == "*":
            return WildNode()
        if string.isdigit():
            return IndexNode(int(string))
        return StringNode(string)

    @staticmethod
    def parse(spec: str) -> list["SpecNode"]:
        return [SpecNode.from_string(part) for part in spec.split(".")]


@dataclass(frozen=True)
class StringNode(SpecNode):
    value: str


@dataclass(frozen=True)
class WildNode(SpecNode):
    pass


@dataclass(frozen=True)
class DoubleWildNode(SpecNode):
    pass


@dataclass(frozen=True)
class IndexNode(SpecNode):
    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Negative indices are not allowed in filter specs.")


def _matches(
    spec: list[SpecNode],
    path: list[pytree.GetAttrKey | pytree.SequenceKey | pytree.DictKey],
) -> bool:
    match spec, path:
        case [], []:
            return True
        case [WildNode(), *s_rest], [_cur, *p_rest]:
            return _matches(s_rest, p_rest)
        case [DoubleWildNode(), *s_rest], [_cur, *p_rest]:
            return _matches(s_rest, p_rest) or _matches(spec, p_rest)
        case [IndexNode(idx), *s_rest], [pytree.SequenceKey(seq_idx), *p_rest]:
            return idx == seq_idx and _matches(s_rest, p_rest)
        case [StringNode(name), *s_rest], [pytree.GetAttrKey(attr) | pytree.DictKey(attr), *p_rest]:
            return name == attr and _matches(s_rest, p_rest)
        case _:
            return False


@dataclass(frozen=True)
class Spec:
    is_exclude: bool
    parts: list[SpecNode]

    @classmethod
    def from_string(cls, raw: str) -> "Spec":
        exclude_prefix = "exclude "
        is_exclude = raw.startswith(exclude_prefix)
        if is_exclude:
            raw = raw[len(exclude_prefix) :]
        return cls(is_exclude=is_exclude, parts=SpecNode.parse(raw))


@dataclass(frozen=True)
class SpecMatch:
    exclude: bool
    match: bool


def _reduce_spec(entries: list[SpecMatch]) -> bool:
    current = False
    for entry in entries:
        if not entry.match:
            continue
        current = not entry.exclude
    return current


def get_filter_spec(tree: PyTree, spec_strs: list[str], filter_type: str) -> PyTree:
    specs = [Spec.from_string(spec) for spec in spec_strs]
    matches = [
        jax.tree_util.tree_map_with_path(
            lambda path, _value: SpecMatch(
                exclude=spec.is_exclude,
                match=_matches(spec.parts, path),
            ),
            tree,
        )
        for spec in specs
    ]
    for raw, match_tree in zip(spec_strs, matches):
        if "index" in raw:
            continue
        any_match = bool(
            jax.tree.reduce(
                lambda a, b: a or b,
                jax.tree.map(lambda entry: entry.match, match_tree),
                initializer=False,
            )
        )
        if not any_match:
            raise ValueError(f"Spec {filter_type} did not match any parameters: {raw}")
    return jax.tree.map(lambda *entries: _reduce_spec(list(entries)), *matches)


def filter_parameters(tree: PyTree, spec_strs: list[str], filter_type: str) -> PyTree:
    spec = get_filter_spec(tree, spec_strs, filter_type)
    return eqx.filter(tree, spec)


def filter_apply_updates(model: PyTree, updates: PyTree) -> PyTree:
    return jax.tree.map(lambda p, u: p + u if u is not None else p, model, updates)


def _path_to_string(path: tuple[Any, ...], sep: str | None = None):
    pieces: list[str] = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            pieces.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            pieces.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            pieces.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            pieces.append(str(key.key))
        else:
            pieces.append(str(key))
    if sep is None:
        return tuple(pieces)
    return sep.join(pieces)


def get_mask_fn(match_name_fn, params):
    return jax.tree_util.tree_map_with_path(
        lambda path, _leaf: match_name_fn(_path_to_string(path, sep="/")),
        params,
    )
