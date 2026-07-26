import pathlib
import sys
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SMACLITE = ROOT / "external" / "smaclite"
if str(SMACLITE) not in sys.path:
    sys.path.insert(0, str(SMACLITE))

from smaclite.env.units.targeters.targeter import LaserBeamTargeter
from smaclite.env.units.unit_command import AttackUnitCommand


class DummyUnit:
    def __init__(self, hp=10, pos=(0, 0)):
        self.hp = hp
        self.pos = np.asarray(pos, dtype=np.float32)
        self.radius = 0.5
        self.radius_sq = self.radius ** 2
        self.plane = "GROUND"
        self.valid_targets = {"GROUND"}
        self.attacking = False
        self.target = None
        self.cooldown = 0
        self.max_cooldown = 1
        self.targeter = types.SimpleNamespace(target=lambda *a, **k: 1.0)

    def has_within_attack_range(self, target):
        return target is not None and target.hp > 0

    def deal_damage(self, target):
        if target is None or target.hp <= 0:
            return 0
        target.hp = max(0, target.hp - 1)
        return 1


class EmptyFinder:
    def query_radius(self, units, radius):
        return [[] for _ in units]


def test_attack_unit_command_stops_for_none_target():
    unit = DummyUnit()
    cmd = AttackUnitCommand(None)
    cmd.clean_up_target(unit)
    assert unit.target is None
    assert cmd.prepare_velocity(unit).shape == (2,)
    assert cmd.execute(unit) == 0


def test_attack_unit_command_stops_for_dead_target():
    unit = DummyUnit()
    target = DummyUnit(hp=0)
    cmd = AttackUnitCommand(target)
    cmd.clean_up_target(unit)
    assert unit.target is None
    assert not unit.attacking
    assert cmd.execute(unit) == 0


def test_laser_noops_for_empty_neighbours():
    origin = DummyUnit(pos=(0, 0))
    target = DummyUnit(pos=(1, 1))
    laser = LaserBeamTargeter(width=1, height=1)
    assert laser.target(origin, target, neighbour_finder=EmptyFinder()) == 0


def test_laser_noops_for_dead_target():
    origin = DummyUnit(pos=(0, 0))
    target = DummyUnit(hp=0, pos=(1, 1))
    laser = LaserBeamTargeter(width=1, height=1)
    assert laser.target(origin, target, neighbour_finder=EmptyFinder()) == 0
