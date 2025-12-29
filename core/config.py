import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Ollama LLM configuration"""
    base_url: str = "http://localhost:11434"
    model: str = "gpt-oss-120b-cloud"
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: int = 4096

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OllamaConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "analytics"
    user: str = "analyst"
    password: str = ""

    def get_connection_string(self, db_type: str = 'postgresql') -> str:
        """Generate connection string"""
        if db_type == 'postgresql':
            return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif db_type == 'mysql':
            return f"mysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        elif db_type == 'sqlite':
            return f"sqlite:///{self.database}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")


@dataclass
class AppConfig:
    """Application configuration"""
    name: str = "Data Analysis Swiss Knife"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    max_upload_size: int = 200  # MB
    session_timeout: int = 3600  # seconds


class ConfigManager:
    """
    Singleton Configuration Manager
    Loads and manages application configuration
    """
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._load_config()
        logger.info("Configuration manager initialized")

    def _load_config(self):
        """Load configuration from multiple sources"""
        # 1. Load from environment variables
        self._load_from_env()

        # 2. Load from config files
        config_paths = [
            Path('config/settings.yaml'),
            Path('config/settings.yml'),
            Path('.config.yaml'),
            Path('.config.yml')
        ]

        for path in config_paths:
            if path.exists():
                self._load_from_file(path)
                break

        # 3. Set defaults if not configured
        self._set_defaults()

    def _load_from_env(self):
        """Load configuration from environment variables"""
        env_mappings = {
            # Ollama
            'OLLAMA_BASE_URL': ('ollama', 'base_url'),
            'OLLAMA_MODEL': ('ollama', 'model'),
            'OLLAMA_TIMEOUT': ('ollama', 'timeout'),
            'OLLAMA_TEMPERATURE': ('ollama', 'temperature'),
            'OLLAMA_MAX_TOKENS': ('ollama', 'max_tokens'),

            # Database
            'POSTGRES_HOST': ('database', 'host'),
            'POSTGRES_PORT': ('database', 'port'),
            'POSTGRES_DB': ('database', 'database'),
            'POSTGRES_USER': ('database', 'user'),
            'POSTGRES_PASSWORD': ('database', 'password'),

            # App
            'APP_NAME': ('app', 'name'),
            'APP_VERSION': ('app', 'version'),
            'DEBUG': ('app', 'debug'),
            'LOG_LEVEL': ('app', 'log_level'),
        }

        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in self._config:
                    self._config[section] = {}

                # Type conversion
                if key in ['port', 'timeout', 'max_tokens', 'max_upload_size', 'session_timeout']:
                    value = int(value)
                elif key in ['temperature']:
                    value = float(value)
                elif key in ['debug']:
                    value = value.lower() in ['true', '1', 'yes']

                self._config[section][key] = value

    def _load_from_file(self, path: Path):
        """Load configuration from YAML file"""
        try:
            with open(path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    # Merge with existing config
                    for section, values in file_config.items():
                        if section not in self._config:
                            self._config[section] = {}
                        self._config[section].update(values)
                    logger.info(f"Loaded configuration from {path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")

    def _set_defaults(self):
        """Set default configuration values"""
        defaults = {
            'ollama': OllamaConfig().__dict__,
            'database': DatabaseConfig().__dict__,
            'app': AppConfig().__dict__
        }

        for section, values in defaults.items():
            if section not in self._config:
                self._config[section] = {}
            for key, value in values.items():
                if key not in self._config[section]:
                    self._config[section][key] = value

    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value

        Args:
            section: Configuration section
            key: Configuration key (optional)
            default: Default value if not found

        Returns:
            Configuration value
        """
        if key is None:
            return self._config.get(section, default)

        section_config = self._config.get(section, {})
        return section_config.get(key, default)

    def set(self, section: str, key: str, value: Any):
        """Set configuration value"""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value

    def get_ollama_config(self) -> OllamaConfig:
        """Get Ollama configuration"""
        return OllamaConfig.from_dict(self._config.get('ollama', {}))

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        db_config = self._config.get('database', {})
        return DatabaseConfig(**{k: v for k, v in db_config.items() if k in DatabaseConfig.__annotations__})

    def get_app_config(self) -> AppConfig:
        """Get application configuration"""
        app_config = self._config.get('app', {})
        return AppConfig(**{k: v for k, v in app_config.items() if k in AppConfig.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return self._config.copy()

    def to_json(self) -> str:
        """Export configuration as JSON"""
        return json.dumps(self._config, indent=2, default=str)

    def save(self, path: Path):
        """Save configuration to file"""
        try:
            with open(path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def reload(self):
        """Reload configuration"""
        self._config.clear()
        self._load_config()
        logger.info("Configuration reloaded")


# Global configuration instance
config = ConfigManager()


# Convenience functions
def get_config(section: str, key: Optional[str] = None, default: Any = None) -> Any:
    """Get configuration value"""
    return config.get(section, key, default)


def get_ollama_config() -> OllamaConfig:
    """Get Ollama configuration"""
    return config.get_ollama_config()


def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return config.get_database_config()


def get_app_config() -> AppConfig:
    """Get application configuration"""
    return config.get_app_config()
