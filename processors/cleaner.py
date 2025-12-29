import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from core.base import DataProcessor, DataContext
import logging

logger = logging.getLogger(__name__)


class MissingValueHandler(DataProcessor):
    """Handle missing values in the dataset"""

    def __init__(self, strategy: str = 'auto', fill_value: Any = None):
        """
        Initialize missing value handler

        Args:
            strategy: 'auto', 'drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'custom'
            fill_value: Custom value for 'custom' strategy
        """
        super().__init__()
        self.strategy = strategy
        self.fill_value = fill_value

    def process(self, context: DataContext) -> DataContext:
        """Process missing values"""
        df = context.data.copy()

        try:
            if self.strategy == 'drop':
                df = df.dropna()
                logger.info(
                    f"Dropped rows with missing values. Remaining: {len(df)}")

            elif self.strategy == 'auto':
                # Auto strategy: numeric -> median, categorical -> mode
                for col in df.columns:
                    if df[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col].fillna(df[col].median(), inplace=True)
                        else:
                            mode_val = df[col].mode()[0] if len(
                                df[col].mode()) > 0 else 'Unknown'
                            df[col].fillna(mode_val, inplace=True)
                logger.info("Applied auto strategy for missing values")

            elif self.strategy == 'mean':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(
                    df[numeric_cols].mean())
                logger.info("Filled missing values with mean")

            elif self.strategy == 'median':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(
                    df[numeric_cols].median())
                logger.info("Filled missing values with median")

            elif self.strategy == 'mode':
                for col in df.columns:
                    if df[col].isnull().any():
                        mode_val = df[col].mode()[0] if len(
                            df[col].mode()) > 0 else 'Unknown'
                        df[col].fillna(mode_val, inplace=True)
                logger.info("Filled missing values with mode")

            elif self.strategy == 'forward_fill':
                df = df.fillna(method='ffill')
                logger.info("Applied forward fill")

            elif self.strategy == 'backward_fill':
                df = df.fillna(method='bfill')
                logger.info("Applied backward fill")

            elif self.strategy == 'custom':
                df = df.fillna(self.fill_value)
                logger.info(
                    f"Filled missing values with custom value: {self.fill_value}")

            context.data = df
            context.metadata['missing_value_handler'] = {
                'strategy': self.strategy,
                'rows_after': len(df)
            }

        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise

        return self._handle_next(context)


class DuplicateRemover(DataProcessor):
    """Remove duplicate rows"""

    def __init__(self, subset: Optional[List[str]] = None, keep: str = 'first'):
        """
        Initialize duplicate remover

        Args:
            subset: Columns to consider for identifying duplicates
            keep: 'first', 'last', or False (remove all duplicates)
        """
        super().__init__()
        self.subset = subset
        self.keep = keep

    def process(self, context: DataContext) -> DataContext:
        """Remove duplicate rows"""
        df = context.data.copy()
        initial_count = len(df)

        try:
            df = df.drop_duplicates(subset=self.subset, keep=self.keep)
            removed_count = initial_count - len(df)

            logger.info(f"Removed {removed_count} duplicate rows")

            context.data = df
            context.metadata['duplicate_remover'] = {
                'removed': removed_count,
                'remaining': len(df)
            }

        except Exception as e:
            logger.error(f"Error removing duplicates: {e}")
            raise

        return self._handle_next(context)


class OutlierHandler(DataProcessor):
    """Handle outliers using various methods"""

    def __init__(self, method: str = 'iqr', threshold: float = 1.5, action: str = 'remove'):
        """
        Initialize outlier handler

        Args:
            method: 'iqr', 'zscore', 'isolation_forest'
            threshold: Threshold for outlier detection
            action: 'remove', 'cap', 'replace'
        """
        super().__init__()
        self.method = method
        self.threshold = threshold
        self.action = action

    def process(self, context: DataContext) -> DataContext:
        """Handle outliers"""
        df = context.data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        try:
            outliers_count = 0

            for col in numeric_cols:
                if self.method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.threshold * IQR
                    upper_bound = Q3 + self.threshold * IQR

                    outliers = (df[col] < lower_bound) | (
                        df[col] > upper_bound)

                elif self.method == 'zscore':
                    from scipy import stats
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outliers = pd.Series(False, index=df.index)
                    outliers.loc[df[col].notna()] = z_scores > self.threshold

                else:
                    continue

                outliers_count += outliers.sum()

                if self.action == 'remove':
                    df = df[~outliers]
                elif self.action == 'cap':
                    df.loc[outliers & (df[col] < lower_bound),
                           col] = lower_bound
                    df.loc[outliers & (df[col] > upper_bound),
                           col] = upper_bound
                elif self.action == 'replace':
                    df.loc[outliers, col] = df[col].median()

            logger.info(
                f"Handled {outliers_count} outliers using {self.method} method")

            context.data = df
            context.metadata['outlier_handler'] = {
                'method': self.method,
                'outliers_found': outliers_count,
                'action': self.action
            }

        except Exception as e:
            logger.error(f"Error handling outliers: {e}")
            raise

        return self._handle_next(context)


class DataTypeConverter(DataProcessor):
    """Convert data types of columns"""

    def __init__(self, conversions: Dict[str, str]):
        """
        Initialize data type converter

        Args:
            conversions: Dictionary mapping column names to target types
                Example: {'age': 'int', 'date': 'datetime', 'category': 'category'}
        """
        super().__init__()
        self.conversions = conversions

    def process(self, context: DataContext) -> DataContext:
        """Convert data types"""
        df = context.data.copy()

        try:
            for col, dtype in self.conversions.items():
                if col not in df.columns:
                    logger.warning(f"Column {col} not found, skipping")
                    continue

                if dtype == 'datetime':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif dtype == 'category':
                    df[col] = df[col].astype('category')
                elif dtype in ['int', 'int64']:
                    df[col] = pd.to_numeric(
                        df[col], errors='coerce').astype('Int64')
                elif dtype in ['float', 'float64']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif dtype == 'string':
                    df[col] = df[col].astype(str)
                else:
                    df[col] = df[col].astype(dtype)

                logger.info(f"Converted {col} to {dtype}")

            context.data = df
            context.metadata['data_type_converter'] = self.conversions

        except Exception as e:
            logger.error(f"Error converting data types: {e}")
            raise

        return self._handle_next(context)


class TextCleaner(DataProcessor):
    """Clean text columns"""

    def __init__(self, operations: List[str] = None):
        """
        Initialize text cleaner

        Args:
            operations: List of operations to perform
                Options: 'lowercase', 'uppercase', 'strip', 'remove_punctuation',
                        'remove_numbers', 'remove_extra_spaces'
        """
        super().__init__()
        self.operations = operations or [
            'strip', 'lowercase', 'remove_extra_spaces']

    def process(self, context: DataContext) -> DataContext:
        """Clean text columns"""
        df = context.data.copy()
        text_cols = df.select_dtypes(include=['object']).columns

        try:
            for col in text_cols:
                if 'lowercase' in self.operations:
                    df[col] = df[col].str.lower()

                if 'uppercase' in self.operations:
                    df[col] = df[col].str.upper()

                if 'strip' in self.operations:
                    df[col] = df[col].str.strip()

                if 'remove_punctuation' in self.operations:
                    df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)

                if 'remove_numbers' in self.operations:
                    df[col] = df[col].str.replace(r'\d+', '', regex=True)

                if 'remove_extra_spaces' in self.operations:
                    df[col] = df[col].str.replace(r'\s+', ' ', regex=True)

            logger.info(f"Cleaned {len(text_cols)} text columns")

            context.data = df
            context.metadata['text_cleaner'] = {
                'operations': self.operations,
                'columns_cleaned': list(text_cols)
            }

        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            raise

        return self._handle_next(context)


class ColumnNameStandardizer(DataProcessor):
    """Standardize column names"""

    def __init__(self, style: str = 'snake_case'):
        """
        Initialize column name standardizer

        Args:
            style: 'snake_case', 'camelCase', 'PascalCase', 'lowercase'
        """
        super().__init__()
        self.style = style

    def process(self, context: DataContext) -> DataContext:
        """Standardize column names"""
        df = context.data.copy()

        try:
            old_columns = df.columns.tolist()

            if self.style == 'snake_case':
                new_columns = [
                    col.lower().replace(' ', '_').replace('-', '_')
                    for col in df.columns
                ]
            elif self.style == 'camelCase':
                new_columns = [
                    ''.join(word.capitalize() if i > 0 else word.lower()
                            for i, word in enumerate(col.split()))
                    for col in df.columns
                ]
            elif self.style == 'PascalCase':
                new_columns = [
                    ''.join(word.capitalize() for word in col.split())
                    for col in df.columns
                ]
            elif self.style == 'lowercase':
                new_columns = [col.lower().replace(' ', '_')
                               for col in df.columns]
            else:
                new_columns = old_columns

            df.columns = new_columns
            logger.info(f"Standardized column names to {self.style}")

            context.data = df
            context.metadata['column_name_standardizer'] = {
                'style': self.style,
                'mapping': dict(zip(old_columns, new_columns))
            }

        except Exception as e:
            logger.error(f"Error standardizing column names: {e}")
            raise

        return self._handle_next(context)


class DataValidator(DataProcessor):
    """Validate data quality"""

    def __init__(self, rules: Dict[str, Any]):
        """
        Initialize data validator

        Args:
            rules: Validation rules
                Example: {
                    'age': {'min': 0, 'max': 120},
                    'email': {'pattern': r'^[\w\.-]+@[\w\.-]+\.\w+$'},
                    'status': {'values': ['active', 'inactive']}
                }
        """
        super().__init__()
        self.rules = rules

    def process(self, context: DataContext) -> DataContext:
        """Validate data"""
        df = context.data.copy()
        validation_results = {}

        try:
            for col, rules in self.rules.items():
                if col not in df.columns:
                    continue

                col_results = {'passed': True, 'issues': []}

                # Min/Max validation
                if 'min' in rules:
                    violations = df[df[col] < rules['min']]
                    if len(violations) > 0:
                        col_results['passed'] = False
                        col_results['issues'].append(
                            f"{len(violations)} values below minimum")

                if 'max' in rules:
                    violations = df[df[col] > rules['max']]
                    if len(violations) > 0:
                        col_results['passed'] = False
                        col_results['issues'].append(
                            f"{len(violations)} values above maximum")

                # Pattern validation
                if 'pattern' in rules:
                    import re
                    violations = ~df[col].astype(
                        str).str.match(rules['pattern'])
                    if violations.sum() > 0:
                        col_results['passed'] = False
                        col_results['issues'].append(
                            f"{violations.sum()} values don't match pattern")

                # Values validation
                if 'values' in rules:
                    violations = ~df[col].isin(rules['values'])
                    if violations.sum() > 0:
                        col_results['passed'] = False
                        col_results['issues'].append(
                            f"{violations.sum()} invalid values")

                validation_results[col] = col_results

            logger.info("Data validation completed")

            context.data = df
            context.metadata['data_validator'] = validation_results

        except Exception as e:
            logger.error(f"Error validating data: {e}")
            raise

        return self._handle_next(context)
