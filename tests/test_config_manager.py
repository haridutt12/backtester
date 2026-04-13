"""
Unit tests for ConfigManager.
"""

import json
import os
import tempfile

import pytest

from src.config.config_manager import ConfigManager


class TestConfigManager:
    def test_loads_default_config_when_file_missing(self, tmp_path):
        cm = ConfigManager(config_path=str(tmp_path / "nonexistent.json"))
        strategies = cm.get_available_strategies()
        assert "MA_CROSSOVER" in strategies

    def test_loads_existing_config(self, tmp_path):
        cfg = {"MY_STRATEGY": {"description": "Test", "short_window": 5}}
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(cfg))
        cm = ConfigManager(config_path=str(cfg_file))
        assert "MY_STRATEGY" in cm.get_available_strategies()

    def test_get_strategy_config_returns_dict(self, tmp_path):
        cm = ConfigManager(config_path=str(tmp_path / "nonexistent.json"))
        cfg = cm.get_strategy_config("MA_CROSSOVER")
        assert isinstance(cfg, dict)
        assert "short_window" in cfg

    def test_get_strategy_config_returns_none_for_missing(self, tmp_path):
        cm = ConfigManager(config_path=str(tmp_path / "nonexistent.json"))
        assert cm.get_strategy_config("DOES_NOT_EXIST") is None

    def test_get_available_strategies_returns_descriptions(self, tmp_path):
        cm = ConfigManager(config_path=str(tmp_path / "nonexistent.json"))
        strategies = cm.get_available_strategies()
        for name, desc in strategies.items():
            assert isinstance(desc, str)

    def test_save_and_reload_strategy(self, tmp_path):
        cfg_path = str(tmp_path / "config.json")
        cm = ConfigManager(config_path=cfg_path)
        cm.save_strategy_config("NEW_STRAT", {"short_window": 7, "description": "New"})
        cm2 = ConfigManager(config_path=cfg_path)
        assert "NEW_STRAT" in cm2.get_available_strategies()
        assert cm2.get_strategy_config("NEW_STRAT")["short_window"] == 7

    def test_invalid_json_falls_back_to_defaults(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ this is not valid JSON }")
        cm = ConfigManager(config_path=str(bad_file))
        assert "MA_CROSSOVER" in cm.get_available_strategies()
