"""Map sampler for Phase 2+ multi-map training.

Phase 2:  fixed / round_robin / seeded_random modes from a Phase 2/3 manifest.
Phase 4:  shuffled_round_robin / uniform_map / uniform_family / weighted / curriculum
          modes from a versioned Phase 4 manifest.
"""
import pathlib
import random
import math
import torch

# UNIFIED_PRIORITY_V1
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import ruamel.yaml as yaml


@dataclass
class MapEntry:
    name: str
    type: str                   # 'builtin' or 'custom'
    path: Optional[str] = None  # required when type='custom'
    family: str = "uncategorised"
    weight: float = 1.0
    map_id: int = 0


_VALID_TYPES = ('builtin', 'custom')


def validate_manifest(manifest_path: str) -> dict:
    """Load and validate a Phase 2/3 map manifest.

    Raises ValueError for:
    - empty map list
    - unknown map type
    - custom entry missing path
    - missing custom map file
    """
    p = pathlib.Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    raw = yaml.YAML(typ='safe').load(p.read_text(encoding='utf-8'))
    if not raw or 'maps' not in raw or not raw['maps']:
        raise ValueError(f"Manifest '{manifest_path}' has no maps.")
    if 'padding' in raw:
        pad = raw['padding']
        for key in ('max_agents', 'max_enemies', 'max_actions', 'max_obs_size'):
            if key not in pad or not isinstance(pad[key], int) or pad[key] <= 0:
                raise ValueError(
                    f"Manifest '{manifest_path}': padding.{key} must be a positive int."
                )
    root = p.parent.parent.parent  # configs/maps -> configs -> project root
    for entry in raw['maps']:
        t = entry.get('type')
        if t not in _VALID_TYPES:
            raise ValueError(
                f"Manifest '{manifest_path}': map '{entry.get('name')}' has "
                f"unknown type {t!r}. Must be one of {_VALID_TYPES}."
            )
        if t == 'custom':
            ep = entry.get('path')
            if not ep:
                raise ValueError(
                    f"Manifest '{manifest_path}': custom map '{entry.get('name')}' "
                    "has no 'path' field."
                )
            abs_path = root / ep
            if not abs_path.exists():
                raise FileNotFoundError(
                    f"Manifest '{manifest_path}': custom map file not found: {abs_path}"
                )
    return raw


def _load_phase4_manifest(manifest_path: str, split: str) -> tuple:
    """Load a Phase 4 versioned manifest and return (raw_dict, list_of_entries).

    Validates version == 1 and that the requested split exists.
    """
    p = pathlib.Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Phase 4 manifest not found: {p}")
    raw = yaml.YAML(typ='safe').load(p.read_text(encoding='utf-8'))
    version = raw.get('version')
    if version != 1:
        raise ValueError(
            f"Phase 4 manifest '{manifest_path}' has version={version!r}. "
            "Expected version: 1. Did you pass a Phase 3 manifest by mistake?"
        )
    splits = raw.get('splits', {})
    if split not in splits:
        available = list(splits.keys())
        raise ValueError(
            f"Phase 4 manifest '{manifest_path}' has no split '{split}'. "
            f"Available splits: {available}"
        )
    root = p.parent
    while not (root / "src").exists() and root != root.parent:
        root = root.parent

    entries = []
    for e in splits[split]:
        ep = e.get('path', '')
        if ep:
            abs_path = root / ep
            if not abs_path.exists():
                raise FileNotFoundError(
                    f"Phase 4 manifest: split='{split}' map '{e.get('name')}' "
                    f"file not found: {abs_path}"
                )
        entries.append(MapEntry(
            name=e['name'],
            type=e.get('type', 'custom'),
            path=ep or None,
            family=e.get('family', 'uncategorised'),
            weight=float(e.get('weight', 1.0)),
            map_id=int(e.get('map_id', 0)),
        ))
    return raw, entries


class MapSampler:
    """Returns the next map entry on each episode reset.

    Phase 2/3 modes (original):
      fixed               — always returns the first map
      round_robin         — cycles through maps in order
      seeded_random       — reproducible random choice

    Phase 4 modes (new):
      shuffled_round_robin — shuffle full list with seeded RNG, iterate once per
                             cycle, then reshuffle. Guarantees each map is visited
                             exactly once per cycle.
      uniform_map          — sample each map with equal probability
      uniform_family       — sample family uniformly, then sample map within family
      weighted             — use per-entry weight field (uniform fallback if all 1.0)
      curriculum           — alias for round_robin (interface for future extension)

    Coverage metrics (available on all modes; updated by next()):
      sampling_cycle             increments each time the full list is exhausted
      maps_seen_this_cycle       resets each cycle
      total_unique_maps_seen     grows until all maps seen once; then saturates
      total_train_maps           len(maps)
      dataset_coverage_fraction  total_unique_maps_seen / total_train_maps

    peek() returns the map that the next next() call would return, without
    advancing the internal index. Used by SMACliteDreamerEnv.__init__ to
    configure the initial env without consuming an episode slot.
    """

    MODES = (
        'fixed', 'round_robin', 'seeded_random',
        'shuffled_round_robin', 'uniform_map', 'uniform_family',
        'weighted', 'curriculum', 'adaptive_priority',
    )

    def __init__(
        self, maps: List[MapEntry], mode: str = 'round_robin', seed: int = 0,
        shared_probabilities=None, shared_version=None,
    ):
        if mode not in self.MODES:
            raise ValueError(
                f"MapSampler mode must be one of {self.MODES}, got {mode!r}"
            )
        if not maps:
            raise ValueError("MapSampler requires at least one MapEntry.")
        self.maps = list(maps)
        self.mode = mode
        self._idx = 0
        self._rng = random.Random(seed)
        self._shared_probabilities = shared_probabilities
        self._shared_version = shared_version

        # seeded_random: pre-generate the first choice so peek() is stable
        self._next_random: Optional[MapEntry] = (
            self._rng.choice(self.maps) if mode == 'seeded_random' else None
        )

        # shuffled_round_robin state
        self._shuffled_order: List[MapEntry] = []
        self._shuffled_idx: int = 0
        if mode in ('shuffled_round_robin', 'curriculum'):
            self._shuffled_order = list(self.maps)
            self._rng.shuffle(self._shuffled_order)

        # uniform_family: group maps by family
        self._by_family: Dict[str, List[MapEntry]] = defaultdict(list)
        for m in self.maps:
            self._by_family[m.family].append(m)
        self._family_list = sorted(self._by_family.keys())

        # weighted: build cumulative weight list
        self._weights = [max(m.weight, 0.0) for m in self.maps]
        total_w = sum(self._weights)
        if total_w <= 0:
            self._weights = [1.0] * len(self.maps)

        # Coverage tracking
        self.sampling_cycle: int = 0
        self.maps_seen_this_cycle: int = 0
        self.total_unique_maps_seen: int = 0
        self._seen_ids: set = set()
        self.total_train_maps: int = len(self.maps)

        # Peek cache for shuffled_round_robin / uniform_family / weighted / uniform_map
        self._peek_cache: Optional[MapEntry] = self._compute_peek()

    def _compute_peek(self) -> Optional[MapEntry]:
        """Compute the peek value for modes that need a pre-generated cache."""
        mode = self.mode
        if mode == 'fixed':
            return self.maps[0]
        if mode == 'round_robin':
            return self.maps[self._idx]
        if mode == 'seeded_random':
            return self._next_random
        if mode in ('shuffled_round_robin', 'curriculum'):
            if not self._shuffled_order:
                return self.maps[0]
            return self._shuffled_order[self._shuffled_idx]
        if mode == 'uniform_map':
            return self._rng.choice(self.maps)
        if mode == 'uniform_family':
            fam = self._rng.choice(self._family_list)
            return self._rng.choice(self._by_family[fam])
        if mode == 'weighted':
            return self._rng.choices(self.maps, weights=self._weights, k=1)[0]
        if mode == 'adaptive_priority':
            return self._rng.choices(self.maps, weights=self._adaptive_weights(), k=1)[0]
        return self.maps[0]

    def _adaptive_weights(self):
        probs = self._shared_probabilities
        if probs is None:
            return [1.0] * len(self.maps)
        try:
            vals = probs.detach().to(dtype=torch.float64, device='cpu').reshape(-1)
        except Exception:
            return [1.0] * len(self.maps)
        if vals.numel() != len(self.maps):
            return [1.0] * len(self.maps)
        vals = torch.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
        if not torch.isfinite(vals).all() or float(vals.sum()) <= 0:
            return [1.0] * len(self.maps)
        return vals.tolist()

    def _update_coverage(self, entry: MapEntry) -> None:
        mid = id(entry)
        self.maps_seen_this_cycle += 1
        if mid not in self._seen_ids:
            self._seen_ids.add(mid)
            self.total_unique_maps_seen = min(
                self.total_unique_maps_seen + 1, self.total_train_maps)

    @property
    def dataset_coverage_fraction(self) -> float:
        if self.total_train_maps == 0:
            return 0.0
        return self.total_unique_maps_seen / self.total_train_maps

    def coverage_metrics(self) -> dict:
        return {
            "sampling_cycle":            self.sampling_cycle,
            "maps_seen_this_cycle":      self.maps_seen_this_cycle,
            "total_unique_maps_seen":    self.total_unique_maps_seen,
            "dataset_coverage_fraction": self.dataset_coverage_fraction,
            "total_train_maps":          self.total_train_maps,
        }

    def peek(self) -> MapEntry:
        """Return the map that the next next() call would return, without advancing."""
        return self._peek_cache

    def next(self) -> MapEntry:
        """Return the next map and advance the internal state."""
        mode = self.mode

        if mode == 'fixed':
            entry = self.maps[0]
            wrapped = False
        elif mode == 'round_robin':
            entry = self.maps[self._idx]
            self._idx = (self._idx + 1) % len(self.maps)
            wrapped = self._idx == 0
        elif mode == 'seeded_random':
            entry = self._next_random
            self._next_random = self._rng.choice(self.maps)
            wrapped = False
        elif mode in ('shuffled_round_robin', 'curriculum'):
            entry = self._shuffled_order[self._shuffled_idx]
            self._shuffled_idx += 1
            wrapped = self._shuffled_idx >= len(self._shuffled_order)
        elif mode == 'uniform_map':
            entry = self._peek_cache
            wrapped = False
        elif mode == 'uniform_family':
            entry = self._peek_cache
            wrapped = False
        elif mode == 'weighted':
            entry = self._peek_cache
            wrapped = False
        elif mode == 'adaptive_priority':
            entry = self._peek_cache
            wrapped = False
        else:
            entry = self.maps[0]
            wrapped = False

        self._update_coverage(entry)
        if wrapped:
            self.sampling_cycle += 1
            self.maps_seen_this_cycle = 0
            if mode in ('shuffled_round_robin', 'curriculum'):
                self._shuffled_order = list(self.maps)
                self._rng.shuffle(self._shuffled_order)
                self._shuffled_idx = 0
        self._peek_cache = self._compute_next_peek(mode)
        return entry

    def advance(self, count: int) -> None:
        """Advance the sampler by ``count`` consumed episodes.

        Used when a recycled environment worker reconstructs a fresh sampler process but
        must continue the same logical per-slot map sequence. Coverage metrics are updated
        exactly as if ``next()`` had been called for each consumed episode, and ``peek()``
        remains the next map that a subsequent reset will consume.
        """
        count = int(count)
        if count < 0:
            raise ValueError(f"advance count must be non-negative, got {count}")
        for _ in range(count):
            self.next()

    def _compute_next_peek(self, mode: str) -> MapEntry:
        """Compute peek after next() has advanced state."""
        if mode == 'fixed':
            return self.maps[0]
        if mode == 'round_robin':
            return self.maps[self._idx]
        if mode == 'seeded_random':
            return self._next_random
        if mode in ('shuffled_round_robin', 'curriculum'):
            return self._shuffled_order[self._shuffled_idx]
        if mode == 'uniform_map':
            return self._rng.choice(self.maps)
        if mode == 'uniform_family':
            fam = self._rng.choice(self._family_list)
            return self._rng.choice(self._by_family[fam])
        if mode == 'weighted':
            return self._rng.choices(self.maps, weights=self._weights, k=1)[0]
        if mode == 'adaptive_priority':
            return self._rng.choices(self.maps, weights=self._adaptive_weights(), k=1)[0]
        return self.maps[0]

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_entries(
        cls,
        entries: List[MapEntry],
        mode: str = 'shuffled_round_robin',
        seed: int = 0,
        shared_probabilities=None,
        shared_version=None,
    ) -> 'MapSampler':
        """Build a sampler directly from a list of MapEntry (no manifest file).

        Used by the multimap factory after folder discovery: each worker reconstructs its
        own sampler from the discovered entries + a worker-offset seed. Thin convenience
        over ``MapSampler(maps=entries, mode=mode, seed=seed)``.
        """
        return cls(
            maps=list(entries), mode=mode, seed=seed,
            shared_probabilities=shared_probabilities,
            shared_version=shared_version,
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str,
        mode: str = 'round_robin',
        seed: int = 0,
    ) -> 'MapSampler':
        """Load and validate a Phase 2/3 manifest, then construct a MapSampler."""
        raw = validate_manifest(manifest_path)
        maps = [
            MapEntry(name=e['name'], type=e['type'], path=e.get('path'))
            for e in raw['maps']
        ]
        return cls(maps=maps, mode=mode, seed=seed)

    @classmethod
    def from_phase4_manifest(
        cls,
        manifest_path: str,
        split: str,
        mode: str = 'shuffled_round_robin',
        seed: int = 42,
    ) -> 'MapSampler':
        """Load a Phase 4 versioned manifest and return a MapSampler for the split.

        Validates version == 1 and that the requested split exists and its map
        files are present on disk.
        """
        _raw, entries = _load_phase4_manifest(manifest_path, split)
        if not entries:
            raise ValueError(
                f"Phase 4 manifest '{manifest_path}' split='{split}' is empty."
            )
        return cls(maps=entries, mode=mode, seed=seed)
