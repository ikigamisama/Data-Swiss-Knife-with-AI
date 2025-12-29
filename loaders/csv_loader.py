import pandas as pd
import io
from pathlib import Path
from typing import Any, Optional, Dict
from ..core.base import DataLoader
import logging

logger = logging.getLogger(__name__)


class CSVLoader(DataLoader):
    """CSV file loader with various encoding and delimiter support"""

    def __init__(self, encoding: str = 'utf-8', delimiter: str = ','):
        self.encoding = encoding
        self.delimiter = delimiter
        self.supported_extensions = ['.csv', '.txt', '.tsv']

    def validate(self, source: Any) -> bool:
        """Validate if source is a valid CSV file"""
        try:
            if isinstance(source, (str, Path)):
                path = Path(source)
                return path.exists() and path.suffix.lower() in self.supported_extensions
            elif hasattr(source, 'read'):
                # File-like object
                return True
            return False
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    def load(self, source: Any, **kwargs) -> pd.DataFrame:
        """
        Load CSV data from various sources

        Args:
            source: File path, file-like object, or URL
            **kwargs: Additional pandas read_csv parameters

        Returns:
            pd.DataFrame: Loaded data
        """
        if not self.validate(source):
            raise ValueError(f"Invalid CSV source: {source}")

        try:
            # Merge default parameters with user-provided ones
            params = {
                'encoding': self.encoding,
                'delimiter': self.delimiter,
                'on_bad_lines': 'skip',
                'low_memory': False
            }
            params.update(kwargs)

            # Handle different source types
            if isinstance(source, (str, Path)):
                logger.info(f"Loading CSV from path: {source}")
                df = pd.read_csv(source, **params)
            elif hasattr(source, 'read'):
                # File-like object (e.g., UploadedFile)
                logger.info("Loading CSV from file-like object")
                source.seek(0)  # Reset to beginning
                df = pd.read_csv(source, **params)
            else:
                raise ValueError(f"Unsupported source type: {type(source)}")

            logger.info(
                f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise RuntimeError(f"Failed to load CSV: {str(e)}")

    def load_chunked(self, source: Any, chunksize: int = 10000, **kwargs):
        """
        Load CSV in chunks for large files

        Args:
            source: File path or file-like object
            chunksize: Number of rows per chunk
            **kwargs: Additional pandas read_csv parameters

        Yields:
            pd.DataFrame: Data chunks
        """
        params = {
            'encoding': self.encoding,
            'delimiter': self.delimiter,
            'chunksize': chunksize
        }
        params.update(kwargs)

        try:
            logger.info(f"Loading CSV in chunks of {chunksize}")
            for chunk in pd.read_csv(source, **params):
                yield chunk
        except Exception as e:
            logger.error(f"Error loading CSV chunks: {e}")
            raise RuntimeError(f"Failed to load CSV chunks: {str(e)}")

    def detect_delimiter(self, source: Any, sample_size: int = 1024) -> str:
        """
        Automatically detect CSV delimiter

        Args:
            source: File path or file-like object
            sample_size: Number of bytes to sample

        Returns:
            str: Detected delimiter
        """
        try:
            if isinstance(source, (str, Path)):
                with open(source, 'r', encoding=self.encoding) as f:
                    sample = f.read(sample_size)
            else:
                source.seek(0)
                sample = source.read(sample_size)
                if isinstance(sample, bytes):
                    sample = sample.decode(self.encoding)
                source.seek(0)

            # Count common delimiters
            delimiters = {
                ',': sample.count(','),
                '\t': sample.count('\t'),
                ';': sample.count(';'),
                '|': sample.count('|')
            }

            detected = max(delimiters, key=delimiters.get)
            logger.info(f"Detected delimiter: '{detected}'")
            return detected

        except Exception as e:
            logger.warning(f"Could not detect delimiter: {e}")
            return ','

    def get_metadata(self, source: Any) -> Dict[str, Any]:
        """
        Get metadata about CSV file without loading all data

        Returns:
            dict: Metadata including row count, column names, etc.
        """
        try:
            # Read just the header
            df_head = pd.read_csv(
                source, nrows=5, encoding=self.encoding, delimiter=self.delimiter)

            # Estimate row count (for files)
            row_count = None
            if isinstance(source, (str, Path)):
                with open(source, 'r', encoding=self.encoding) as f:
                    row_count = sum(1 for _ in f) - 1  # Subtract header

            metadata = {
                'columns': df_head.columns.tolist(),
                'column_count': len(df_head.columns),
                'estimated_rows': row_count,
                'dtypes': df_head.dtypes.to_dict(),
                'sample': df_head.to_dict('records')
            }

            return metadata

        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
            return {}


class TSVLoader(CSVLoader):
    """Tab-separated values loader"""

    def __init__(self, encoding: str = 'utf-8'):
        super().__init__(encoding=encoding, delimiter='\t')
        self.supported_extensions = ['.tsv', '.tab']


class ExcelLoader(DataLoader):
    """Excel file loader"""

    def __init__(self):
        self.supported_extensions = ['.xlsx', '.xls', '.xlsm']

    def validate(self, source: Any) -> bool:
        """Validate if source is a valid Excel file"""
        try:
            if isinstance(source, (str, Path)):
                path = Path(source)
                return path.exists() and path.suffix.lower() in self.supported_extensions
            elif hasattr(source, 'read'):
                return True
            return False
        except Exception:
            return False

    def load(self, source: Any, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load Excel data

        Args:
            source: File path or file-like object
            sheet_name: Sheet name to load (default: first sheet)
            **kwargs: Additional pandas read_excel parameters

        Returns:
            pd.DataFrame: Loaded data
        """
        if not self.validate(source):
            raise ValueError(f"Invalid Excel source: {source}")

        try:
            params = {'sheet_name': sheet_name or 0}
            params.update(kwargs)

            if isinstance(source, (str, Path)):
                logger.info(f"Loading Excel from path: {source}")
                df = pd.read_excel(source, **params)
            elif hasattr(source, 'read'):
                logger.info("Loading Excel from file-like object")
                source.seek(0)
                df = pd.read_excel(source, **params)
            else:
                raise ValueError(f"Unsupported source type: {type(source)}")

            logger.info(
                f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Error loading Excel: {e}")
            raise RuntimeError(f"Failed to load Excel: {str(e)}")

    def get_sheet_names(self, source: Any) -> list:
        """Get list of sheet names in Excel file"""
        try:
            if isinstance(source, (str, Path)):
                xl_file = pd.ExcelFile(source)
            else:
                source.seek(0)
                xl_file = pd.ExcelFile(source)

            return xl_file.sheet_names
        except Exception as e:
            logger.error(f"Error getting sheet names: {e}")
            return []

    def load_all_sheets(self, source: Any, **kwargs) -> Dict[str, pd.DataFrame]:
        """Load all sheets from Excel file"""
        try:
            sheet_names = self.get_sheet_names(source)
            return {
                sheet: self.load(source, sheet_name=sheet, **kwargs)
                for sheet in sheet_names
            }
        except Exception as e:
            logger.error(f"Error loading all sheets: {e}")
            raise RuntimeError(f"Failed to load all sheets: {str(e)}")


class JSONLoader(DataLoader):
    """JSON file loader"""

    def __init__(self):
        self.supported_extensions = ['.json', '.jsonl']

    def validate(self, source: Any) -> bool:
        """Validate if source is a valid JSON file"""
        try:
            if isinstance(source, (str, Path)):
                path = Path(source)
                return path.exists() and path.suffix.lower() in self.supported_extensions
            elif hasattr(source, 'read'):
                return True
            return False
        except Exception:
            return False

    def load(self, source: Any, orient: str = 'records', **kwargs) -> pd.DataFrame:
        """
        Load JSON data

        Args:
            source: File path or file-like object
            orient: JSON orientation (records, index, columns, values)
            **kwargs: Additional pandas read_json parameters

        Returns:
            pd.DataFrame: Loaded data
        """
        if not self.validate(source):
            raise ValueError(f"Invalid JSON source: {source}")

        try:
            params = {'orient': orient}
            params.update(kwargs)

            if isinstance(source, (str, Path)):
                logger.info(f"Loading JSON from path: {source}")
                df = pd.read_json(source, **params)
            elif hasattr(source, 'read'):
                logger.info("Loading JSON from file-like object")
                source.seek(0)
                df = pd.read_json(source, **params)
            else:
                raise ValueError(f"Unsupported source type: {type(source)}")

            logger.info(
                f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise RuntimeError(f"Failed to load JSON: {str(e)}")


class ParquetLoader(DataLoader):
    """Parquet file loader"""

    def __init__(self):
        self.supported_extensions = ['.parquet', '.pq']

    def validate(self, source: Any) -> bool:
        """Validate if source is a valid Parquet file"""
        try:
            if isinstance(source, (str, Path)):
                path = Path(source)
                return path.exists() and path.suffix.lower() in self.supported_extensions
            elif hasattr(source, 'read'):
                return True
            return False
        except Exception:
            return False

    def load(self, source: Any, **kwargs) -> pd.DataFrame:
        """
        Load Parquet data

        Args:
            source: File path or file-like object
            **kwargs: Additional pandas read_parquet parameters

        Returns:
            pd.DataFrame: Loaded data
        """
        if not self.validate(source):
            raise ValueError(f"Invalid Parquet source: {source}")

        try:
            if isinstance(source, (str, Path)):
                logger.info(f"Loading Parquet from path: {source}")
                df = pd.read_parquet(source, **kwargs)
            elif hasattr(source, 'read'):
                logger.info("Loading Parquet from file-like object")
                source.seek(0)
                df = pd.read_parquet(source, **kwargs)
            else:
                raise ValueError(f"Unsupported source type: {type(source)}")

            logger.info(
                f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Error loading Parquet: {e}")
            raise RuntimeError(f"Failed to load Parquet: {str(e)}")
