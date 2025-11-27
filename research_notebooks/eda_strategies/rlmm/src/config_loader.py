"""
Configuration Loader Module

Handles loading, validation, and merging of YAML configuration files
for the Market Diagnostic Report system.
"""

import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
import copy


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def load_config(config_path: str, profile_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file and optionally merge with a profile.

    Args:
        config_path: Path to the main YAML configuration file
        profile_path: Optional path to a profile-specific YAML file

    Returns:
        Dictionary with the complete merged configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ConfigValidationError: If configuration is invalid
        yaml.YAMLError: If YAML parsing fails
    """
    # Load base config
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ConfigValidationError(f"Configuration file is empty: {config_path}")

    # Load and merge profile if provided
    if profile_path:
        profile_file = Path(profile_path)
        if not profile_file.exists():
            raise FileNotFoundError(f"Profile file not found: {profile_path}")

        with open(profile_file, 'r') as f:
            profile = yaml.safe_load(f)

        if profile is not None:
            config = merge_configs(config, profile)

    # Validate config
    validate_config(config)

    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    Override values take precedence over base values.

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that the configuration has all required fields and valid values.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if valid

    Raises:
        ConfigValidationError: If validation fails
    """
    required_sections = [
        'metadata',
        'data',
        'windows',
        'thresholds',
        'scoring',
        'labels',
        'strategy_profiles',
        'plotting',
        'pdf'
    ]

    # Check required top-level sections
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ConfigValidationError(
            f"Missing required configuration sections: {', '.join(missing_sections)}"
        )

    # Validate metadata
    _validate_metadata(config.get('metadata', {}))

    # Validate data parameters
    _validate_data_params(config.get('data', {}))

    # Validate windows
    _validate_windows(config.get('windows', {}))

    # Validate thresholds
    _validate_thresholds(config.get('thresholds', {}))

    # Validate scoring
    _validate_scoring(config.get('scoring', {}))

    # Validate strategy profiles
    _validate_strategy_profiles(config.get('strategy_profiles', {}))

    # Validate plotting
    _validate_plotting(config.get('plotting', {}))

    return True


def _validate_metadata(metadata: Dict[str, Any]) -> None:
    """Validate metadata section."""
    required_fields = ['version']
    missing = [f for f in required_fields if f not in metadata]
    if missing:
        raise ConfigValidationError(f"Missing metadata fields: {', '.join(missing)}")


def _validate_data_params(data: Dict[str, Any]) -> None:
    """Validate data parameters section."""
    required_subsections = ['candles', 'orderbook', 'trades']
    missing = [s for s in required_subsections if s not in data]
    if missing:
        raise ConfigValidationError(f"Missing data subsections: {', '.join(missing)}")

    # Validate candles
    if 'timeframe' not in data['candles']:
        raise ConfigValidationError("Missing 'timeframe' in data.candles")

    # Validate orderbook
    if 'levels_to_analyze' not in data['orderbook']:
        raise ConfigValidationError("Missing 'levels_to_analyze' in data.orderbook")
    if not isinstance(data['orderbook']['levels_to_analyze'], int) or data['orderbook']['levels_to_analyze'] < 1:
        raise ConfigValidationError("'levels_to_analyze' must be a positive integer")


def _validate_windows(windows: Dict[str, Any]) -> None:
    """Validate windows section."""
    required_subsections = ['volatility', 'spread', 'orderbook_depth', 'trade_flow']
    missing = [s for s in required_subsections if s not in windows]
    if missing:
        raise ConfigValidationError(f"Missing window subsections: {', '.join(missing)}")


def _validate_thresholds(thresholds: Dict[str, Any]) -> None:
    """Validate thresholds section."""
    required_subsections = ['volatility', 'spread', 'orderbook', 'trend', 'trade_flow']
    missing = [s for s in required_subsections if s not in thresholds]
    if missing:
        raise ConfigValidationError(f"Missing threshold subsections: {', '.join(missing)}")


def _validate_scoring(scoring: Dict[str, Any]) -> None:
    """Validate scoring section."""
    if 'weights' not in scoring:
        raise ConfigValidationError("Missing 'weights' in scoring section")

    weights = scoring['weights']
    required_weights = [
        'volatility_score',
        'spread_score',
        'depth_score',
        'stability_score',
        'predictability_score'
    ]

    missing = [w for w in required_weights if w not in weights]
    if missing:
        raise ConfigValidationError(f"Missing scoring weights: {', '.join(missing)}")

    # Validate weights sum to 1.0 (with tolerance)
    total_weight = sum(weights[w] for w in required_weights)
    if abs(total_weight - 1.0) > 0.01:
        raise ConfigValidationError(
            f"Scoring weights must sum to 1.0, got {total_weight}"
        )


def _validate_strategy_profiles(profiles: Dict[str, Any]) -> None:
    """Validate strategy profiles section."""
    if not profiles:
        raise ConfigValidationError("At least one strategy profile is required")

    for profile_name, profile_config in profiles.items():
        if 'description' not in profile_config:
            raise ConfigValidationError(
                f"Strategy profile '{profile_name}' missing 'description'"
            )


def _validate_plotting(plotting: Dict[str, Any]) -> None:
    """Validate plotting section."""
    if 'theme' not in plotting:
        raise ConfigValidationError("Missing 'theme' in plotting section")

    if 'colors' not in plotting:
        raise ConfigValidationError("Missing 'colors' in plotting section")

    if 'export' not in plotting:
        raise ConfigValidationError("Missing 'export' in plotting section")


def get_nested(config: Dict, path: str, default: Any = None) -> Any:
    """
    Access nested configuration values using dot notation.

    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., 'thresholds.volatility.high')
        default: Default value if path not found

    Returns:
        Value at the specified path or default if not found

    Examples:
        >>> config = {'a': {'b': {'c': 42}}}
        >>> get_nested(config, 'a.b.c')
        42
        >>> get_nested(config, 'a.b.x', default=0)
        0
    """
    keys = path.split('.')
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def get_threshold_value(config: Dict, category: str, level: str) -> float:
    """
    Convenience function to get threshold values.

    Args:
        config: Configuration dictionary
        category: Threshold category (e.g., 'volatility', 'spread')
        level: Threshold level (e.g., 'high', 'low')

    Returns:
        Threshold value

    Example:
        >>> get_threshold_value(config, 'volatility', 'high')
        0.005
    """
    return get_nested(config, f'thresholds.{category}.{level}', default=0.0)


def get_window_value(config: Dict, category: str, window_type: str) -> int:
    """
    Convenience function to get window values.

    Args:
        config: Configuration dictionary
        category: Window category (e.g., 'volatility', 'spread')
        window_type: Window type (e.g., 'short', 'medium', 'long')

    Returns:
        Window size

    Example:
        >>> get_window_value(config, 'volatility', 'short')
        15
    """
    return get_nested(config, f'windows.{category}.{window_type}', default=30)


def get_label(config: Dict, category: str, key: str) -> str:
    """
    Get a label from the configuration.

    Args:
        config: Configuration dictionary
        category: Label category (e.g., 'volatility', 'spread')
        key: Label key (e.g., 'high', 'low')

    Returns:
        Label text

    Example:
        >>> get_label(config, 'volatility', 'high')
        'Alta'
    """
    return get_nested(config, f'labels.{category}.{key}', default=key.upper())


def get_color(config: Dict, color_name: str) -> str:
    """
    Get a color from the configuration.

    Args:
        config: Configuration dictionary
        color_name: Color name (e.g., 'primary', 'success')

    Returns:
        Color hex code

    Example:
        >>> get_color(config, 'primary')
        '#1f77b4'
    """
    return get_nested(config, f'plotting.colors.{color_name}', default='#000000')
