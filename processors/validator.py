import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from core.base import DataProcessor, DataContext
from core.exceptions import DataValidationError
import re
import logging

logger = logging.getLogger(__name__)


class SchemaValidator(DataProcessor):
    """Validate data against a schema"""

    def __init__(self, schema: Dict[str, Dict[str, Any]], strict: bool = False):
        """
        Initialize schema validator

        Args:
            schema: Schema definition
                Example: {
                    'age': {'type': 'int', 'min': 0, 'max': 120, 'required': True},
                    'email': {'type': 'string', 'pattern': r'^[\w\.-]+@[\w\.-]+\.\w+$'},
                    'status': {'type': 'string', 'values': ['active', 'inactive']}
                }
            strict: If True, raise error on validation failure
        """
        super().__init__()
        self.schema = schema
        self.strict = strict
        self.validation_errors: List[Dict[str, Any]] = []

    def process(self, context: DataContext) -> DataContext:
        """Validate data against schema"""
        df = context.data.copy()
        self.validation_errors = []

        try:
            for column, rules in self.schema.items():
                # Check if column exists
                if column not in df.columns:
                    if rules.get('required', False):
                        self._add_error('missing_column', column,
                                        f"Required column '{column}' not found")
                    continue

                # Type validation
                if 'type' in rules:
                    self._validate_type(df, column, rules['type'])

                # Range validation
                if 'min' in rules:
                    self._validate_min(df, column, rules['min'])
                if 'max' in rules:
                    self._validate_max(df, column, rules['max'])

                # Pattern validation
                if 'pattern' in rules:
                    self._validate_pattern(df, column, rules['pattern'])

                # Values validation
                if 'values' in rules:
                    self._validate_values(df, column, rules['values'])

                # Uniqueness validation
                if rules.get('unique', False):
                    self._validate_unique(df, column)

                # Custom validation
                if 'validator' in rules:
                    self._validate_custom(df, column, rules['validator'])

            # Handle validation errors
            if self.validation_errors:
                logger.warning(
                    f"Found {len(self.validation_errors)} validation errors")
                if self.strict:
                    raise DataValidationError(
                        f"Validation failed: {len(self.validation_errors)} errors")

            context.data = df
            context.metadata['schema_validator'] = {
                'errors': self.validation_errors,
                'passed': len(self.validation_errors) == 0
            }

        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            raise

        return self._handle_next(context)

    def _add_error(self, error_type: str, column: str, message: str):
        """Add validation error"""
        self.validation_errors.append({
            'type': error_type,
            'column': column,
            'message': message
        })

    def _validate_type(self, df: pd.DataFrame, column: str, expected_type: str):
        """Validate column type"""
        actual_type = df[column].dtype

        type_mapping = {
            'int': np.integer,
            'float': np.floating,
            'string': object,
            'datetime': np.datetime64,
            'bool': bool
        }

        expected = type_mapping.get(expected_type)
        if expected and not np.issubdtype(actual_type, expected):
            self._add_error('type_mismatch', column,
                            f"Expected {expected_type}, got {actual_type}")

    def _validate_min(self, df: pd.DataFrame, column: str, min_value: Any):
        """Validate minimum value"""
        violations = df[df[column] < min_value]
        if len(violations) > 0:
            self._add_error('min_violation', column,
                            f"{len(violations)} values below minimum {min_value}")

    def _validate_max(self, df: pd.DataFrame, column: str, max_value: Any):
        """Validate maximum value"""
        violations = df[df[column] > max_value]
        if len(violations) > 0:
            self._add_error('max_violation', column,
                            f"{len(violations)} values above maximum {max_value}")

    def _validate_pattern(self, df: pd.DataFrame, column: str, pattern: str):
        """Validate string pattern"""
        violations = ~df[column].astype(str).str.match(pattern, na=False)
        if violations.sum() > 0:
            self._add_error('pattern_violation', column,
                            f"{violations.sum()} values don't match pattern")

    def _validate_values(self, df: pd.DataFrame, column: str, valid_values: List[Any]):
        """Validate against allowed values"""
        violations = ~df[column].isin(valid_values)
        if violations.sum() > 0:
            self._add_error('value_violation', column,
                            f"{violations.sum()} invalid values")

    def _validate_unique(self, df: pd.DataFrame, column: str):
        """Validate uniqueness"""
        duplicates = df[column].duplicated().sum()
        if duplicates > 0:
            self._add_error('uniqueness_violation', column,
                            f"{duplicates} duplicate values found")

    def _validate_custom(self, df: pd.DataFrame, column: str, validator: Callable):
        """Apply custom validation function"""
        try:
            result = validator(df[column])
            if not result:
                self._add_error('custom_validation', column,
                                "Custom validation failed")
        except Exception as e:
            self._add_error('custom_validation', column, str(e))


class DataQualityValidator(DataProcessor):
    """Validate data quality metrics"""

    def __init__(self, thresholds: Dict[str, float]):
        """
        Initialize data quality validator

        Args:
            thresholds: Quality thresholds
                Example: {
                    'completeness': 0.95,  # Max 5% missing
                    'uniqueness': 0.90,    # Min 90% unique in key columns
                    'consistency': 0.95    # Min 95% consistent formats
                }
        """
        super().__init__()
        self.thresholds = thresholds
        self.quality_metrics: Dict[str, float] = {}

    def process(self, context: DataContext) -> DataContext:
        """Validate data quality"""
        df = context.data.copy()

        try:
            # Calculate quality metrics
            total_cells = df.shape[0] * df.shape[1]

            # Completeness
            missing_cells = df.isnull().sum().sum()
            completeness = 1 - (missing_cells / total_cells)
            self.quality_metrics['completeness'] = completeness

            # Check threshold
            if 'completeness' in self.thresholds:
                if completeness < self.thresholds['completeness']:
                    logger.warning(
                        f"Completeness {completeness:.2%} below threshold "
                        f"{self.thresholds['completeness']:.2%}"
                    )

            # Accuracy (check for outliers)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                from scipy import stats
                outlier_count = 0
                for col in numeric_cols:
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outlier_count += (z_scores > 3).sum()

                accuracy = 1 - (outlier_count /
                                (df.shape[0] * len(numeric_cols)))
                self.quality_metrics['accuracy'] = accuracy

            logger.info(f"Data quality validated: {self.quality_metrics}")

            context.data = df
            context.metadata['quality_validator'] = {
                'metrics': self.quality_metrics,
                'thresholds': self.thresholds
            }

        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            raise

        return self._handle_next(context)


class BusinessRuleValidator(DataProcessor):
    """Validate business rules"""

    def __init__(self, rules: List[Dict[str, Any]]):
        """
        Initialize business rule validator

        Args:
            rules: List of business rules
                Example: [
                    {
                        'name': 'price_positive',
                        'condition': lambda df: df['price'] > 0,
                        'message': 'Price must be positive'
                    },
                    {
                        'name': 'end_after_start',
                        'condition': lambda df: df['end_date'] > df['start_date'],
                        'message': 'End date must be after start date'
                    }
                ]
        """
        super().__init__()
        self.rules = rules
        self.violations: List[Dict[str, Any]] = []

    def process(self, context: DataContext) -> DataContext:
        """Validate business rules"""
        df = context.data.copy()
        self.violations = []

        try:
            for rule in self.rules:
                name = rule['name']
                condition = rule['condition']
                message = rule.get('message', f"Rule '{name}' violated")

                try:
                    # Apply condition
                    mask = condition(df)
                    violations = (~mask).sum()

                    if violations > 0:
                        self.violations.append({
                            'rule': name,
                            'violations': int(violations),
                            'message': message
                        })
                        logger.warning(
                            f"Rule '{name}' violated {violations} times")

                except Exception as e:
                    logger.error(f"Error checking rule '{name}': {e}")

            context.data = df
            context.metadata['business_rule_validator'] = {
                'violations': self.violations,
                'rules_checked': len(self.rules)
            }

        except Exception as e:
            logger.error(f"Business rule validation failed: {e}")
            raise

        return self._handle_next(context)


class ReferentialIntegrityValidator(DataProcessor):
    """Validate referential integrity between datasets"""

    def __init__(self, foreign_keys: Dict[str, tuple]):
        """
        Initialize referential integrity validator

        Args:
            foreign_keys: Dict mapping column to (reference_df, reference_column)
                Example: {
                    'customer_id': (customers_df, 'id'),
                    'product_id': (products_df, 'id')
                }
        """
        super().__init__()
        self.foreign_keys = foreign_keys
        self.orphaned_records: Dict[str, int] = {}

    def process(self, context: DataContext) -> DataContext:
        """Validate referential integrity"""
        df = context.data.copy()

        try:
            for fk_column, (ref_df, ref_column) in self.foreign_keys.items():
                if fk_column not in df.columns:
                    logger.warning(
                        f"Foreign key column '{fk_column}' not found")
                    continue

                # Find orphaned records
                orphaned = ~df[fk_column].isin(ref_df[ref_column])
                orphaned_count = orphaned.sum()

                if orphaned_count > 0:
                    self.orphaned_records[fk_column] = int(orphaned_count)
                    logger.warning(
                        f"Found {orphaned_count} orphaned records in '{fk_column}'"
                    )

            context.data = df
            context.metadata['referential_integrity'] = {
                'orphaned_records': self.orphaned_records
            }

        except Exception as e:
            logger.error(f"Referential integrity validation failed: {e}")
            raise

        return self._handle_next(context)
