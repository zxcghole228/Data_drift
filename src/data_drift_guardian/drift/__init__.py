"""Статистические методы обнаружения дрейфа."""

from .numeric import ks_test, wasserstein

__all__ = ["ks_test", "wasserstein"]
