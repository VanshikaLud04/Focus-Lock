"""Tests for AdaptiveSampler."""
import time
import numpy as np
import pytest
from focuslock.detection.sampler import AdaptiveSampler


@pytest.fixture
def sampler():
    cfg = {"enabled": True, "min_interval_sec": 0.1, "max_interval_sec": 1.0, "ema_alpha": 0.5}
    return AdaptiveSampler(cfg)


def black_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def noisy_frame():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


def test_first_frame_always_infers(sampler):
    should, _ = sampler.tick(black_frame(), None)
    assert should is True


def test_disabled_sampler_always_infers():
    cfg = {"enabled": False, "min_interval_sec": 0.1, "max_interval_sec": 1.0, "ema_alpha": 0.5}
    s = AdaptiveSampler(cfg)
    for _ in range(5):
        should, _ = s.tick(black_frame(), black_frame())
        assert should is True


def test_interval_bounds(sampler):
    for _ in range(20):
        sampler.tick(noisy_frame(), black_frame())
    assert sampler._min_int <= sampler.current_interval_sec <= sampler._max_int


def test_returns_interval_ms(sampler):
    _, ms = sampler.tick(black_frame(), None)
    assert isinstance(ms, float)
    assert ms > 0


def test_static_scene_slows_down(sampler):
    """After many identical frames, the interval should increase."""
    frame = black_frame()
    for _ in range(30):
        sampler.tick(frame.copy(), frame.copy())
        time.sleep(0.01)
    # Interval should be well above minimum
    assert sampler.current_interval_sec > sampler._min_int * 2


def test_motion_entropy_static():
    diff = np.zeros((480, 640, 3), dtype=np.uint8)
    e    = AdaptiveSampler._motion_entropy(diff)
    assert e < 1.0   # low entropy for static scene


def test_motion_entropy_noisy():
    diff = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    e    = AdaptiveSampler._motion_entropy(diff)
    assert e > 2.0   # high entropy for noisy scene
