from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """Abstract base class for all data loaders using Strategy Pattern"""

    def __init__(self):
        self.metadata: Dict[str, Any] = {}
        self.last_loaded: Optional[pd.DataFrame] = None

    @abstractmethod
    def validate(self, source: Any) -> bool:
        """
        Validate if the source is compatible with this loader

        Args:
            source: Data source (file path, URL, connection string, etc.)

        Returns:
            True if source is valid, False otherwise
        """
        pass

    @abstractmethod
    def load(self, source: Any, **kwargs) -> pd.DataFrame:
        """
        Load data from source

        Args:
            source: Data source
            **kwargs: Additional loader-specific parameters

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            DataLoadError: If loading fails
        """
        pass

    def can_handle(self, source: Any) -> bool:
        """
        Check if this loader can handle the given source

        Args:
            source: Data source to check

        Returns:
            True if loader can handle this source
        """
        return self.validate(source)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the last loaded data

        Returns:
            Dictionary containing metadata
        """
        return self.metadata.copy()

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats/sources

        Returns:
            List of supported formats
        """
        return []

    def preview(self, source: Any, n_rows: int = 5, **kwargs) -> pd.DataFrame:
        """
        Preview data without loading everything

        Args:
            source: Data source
            n_rows: Number of rows to preview
            **kwargs: Additional parameters

        Returns:
            pd.DataFrame: Preview of data
        """
        df = self.load(source, **kwargs)
        return df.head(n_rows)

    def get_info(self, source: Any) -> Dict[str, Any]:
        """
        Get information about data source without loading

        Args:
            source: Data source

        Returns:
            Dictionary with source information
        """
        return {
            'source': str(source),
            'loader': self.__class__.__name__,
            'supported_formats': self.get_supported_formats()
        }


class LoaderRegistry:
    """Registry for managing multiple data loaders"""

    def __init__(self):
        self._loaders: Dict[str, BaseDataLoader] = {}

    def register(self, name: str, loader: BaseDataLoader) -> None:
        """
        Register a loader

        Args:
            name: Loader name
            loader: Loader instance
        """
        self._loaders[name] = loader
        logger.info(f"Registered loader: {name}")

    def unregister(self, name: str) -> None:
        """
        Unregister a loader

        Args:
            name: Loader name
        """
        if name in self._loaders:
            del self._loaders[name]
            logger.info(f"Unregistered loader: {name}")

    def get_loader(self, name: str) -> Optional[BaseDataLoader]:
        """
        Get a specific loader by name

        Args:
            name: Loader name

        Returns:
            Loader instance or None
        """
        return self._loaders.get(name)

    def find_loader(self, source: Any) -> Optional[BaseDataLoader]:
        """
        Find appropriate loader for given source

        Args:
            source: Data source

        Returns:
            First compatible loader or None
        """
        for loader in self._loaders.values():
            if loader.can_handle(source):
                logger.info(
                    f"Found compatible loader: {loader.__class__.__name__}")
                return loader

        logger.warning(f"No compatible loader found for source: {source}")
        return None

    def load(self, source: Any, loader_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data using appropriate loader

        Args:
            source: Data source
            loader_name: Specific loader to use (optional)
            **kwargs: Additional parameters

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            ValueError: If no compatible loader found
        """
        if loader_name:
            loader = self.get_loader(loader_name)
            if not loader:
                raise ValueError(f"Loader not found: {loader_name}")
        else:
            loader = self.find_loader(source)
            if not loader:
                raise ValueError(f"No compatible loader found for: {source}")

        return loader.load(source, **kwargs)

    def list_loaders(self) -> List[str]:
        """
        List all registered loaders

        Returns:
            List of loader names
        """
        return list(self._loaders.keys())


class AutoLoader:
    """Automatically select and use appropriate loader"""

    def __init__(self):
        self.registry = LoaderRegistry()
        self._setup_default_loaders()

    def _setup_default_loaders(self):
        """Setup default loaders"""
        try:
            from .csv_loader import CSVLoader, ExcelLoader, JSONLoader, ParquetLoader

            self.registry.register('csv', CSVLoader())
            self.registry.register('excel', ExcelLoader())
            self.registry.register('json', JSONLoader())
            self.registry.register('parquet', ParquetLoader())

            logger.info("Default loaders registered")
        except ImportError as e:
            logger.warning(f"Could not register default loaders: {e}")

    def load(self, source: Any, **kwargs) -> pd.DataFrame:
        """
        Automatically detect and load data

        Args:
            source: Data source (path, URL, etc.)
            **kwargs: Additional parameters

        Returns:
            pd.DataFrame: Loaded data
        """
        # Try to detect from file extension
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            if source_path.exists():
                extension = source_path.suffix.lower()

                loader_map = {
                    '.csv': 'csv',
                    '.xlsx': 'excel',
                    '.xls': 'excel',
                    '.json': 'json',
                    '.parquet': 'parquet',
                    '.pq': 'parquet'
                }

                loader_name = loader_map.get(extension)
                if loader_name:
                    logger.info(f"Auto-detected loader: {loader_name}")
                    return self.registry.load(source, loader_name, **kwargs)

        # Try to find compatible loader
        return self.registry.load(source, **kwargs)

    def add_loader(self, name: str, loader: BaseDataLoader):
        """Add custom loader to registry"""
        self.registry.register(name, loader)


# Singleton instance
_auto_loader_instance = None


def get_auto_loader() -> AutoLoader:
    """Get singleton AutoLoader instance"""
    global _auto_loader_instance
    if _auto_loader_instance is None:
        _auto_loader_instance = AutoLoader()
    return _auto_loader_instance
