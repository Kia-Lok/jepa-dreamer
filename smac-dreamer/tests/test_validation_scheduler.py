import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
R2 = ROOT / "external" / "r2dreamer"
if str(R2) not in sys.path:
    sys.path.insert(0, str(R2))

from smacdreamer.validation_trainer import ValidationEvery


def test_run_at_start_false_skips_step_zero():
    sched = ValidationEvery(100, run_at_start=False, initial_step=0)
    assert sched(0) == 0
    assert sched(99) == 0
    assert sched(100) == 1
    assert sched(101) == 0
    assert sched(200) == 1


def test_run_at_start_true_evaluates_step_zero_then_intervals():
    sched = ValidationEvery(100, run_at_start=True, initial_step=0)
    assert sched(0) == 1
    assert sched(1) == 0
    assert sched(100) == 1
    assert sched(200) == 1


def test_resume_at_boundary_does_not_duplicate_last_validation():
    sched = ValidationEvery(100, run_at_start=False, initial_step=200)
    assert sched(200) == 0
    assert sched(299) == 0
    assert sched(300) == 1


def test_resume_near_boundary_keeps_next_interval():
    sched = ValidationEvery(100, run_at_start=False, initial_step=199)
    assert sched(199) == 0
    assert sched(200) == 1
    assert sched(201) == 0
