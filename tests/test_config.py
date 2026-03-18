"""Tests for config.py — DiscountConfig validation."""

import pytest

from config import DiscountConfig


class TestDiscountConfigValidation:
    """DiscountConfig.__post_init__ の assert 検証."""

    def test_valid_config_does_not_raise(self):
        DiscountConfig(deliver_ramp_step=10, deliver_limit_step=100)

    def test_ramp_equal_limit_raises(self):
        with pytest.raises(AssertionError):
            DiscountConfig(deliver_ramp_step=100, deliver_limit_step=100)

    def test_ramp_greater_than_limit_raises(self):
        with pytest.raises(AssertionError):
            DiscountConfig(deliver_ramp_step=101, deliver_limit_step=100)

    def test_ramp_zero_is_valid(self):
        DiscountConfig(deliver_ramp_step=0, deliver_limit_step=1)
