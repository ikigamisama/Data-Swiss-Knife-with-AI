import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from core.base import DataProcessor, DataContext
import logging

logger = logging.getLogger(__name__)


class DataAggregator(DataProcessor):
    """Aggregate data using various methods"""

    def __init__(self, group_by: List[str], aggregations: Dict[str, Any]):
        """
        Initialize data aggregator

        Args:
            group_by: Columns to group by
            aggregations: Dictionary mapping columns to aggregation functions
                Example: {
                    'sales': ['sum', 'mean', 'count'],
                    'quantity': 'sum',
                    'price': lambda x: x.max() - x.min()
                }
        """
        super().__init__()
        self.group_by = group_by
        self.aggregations = aggregations

    def process(self, context: DataContext) -> DataContext:
        """Perform aggregation"""
        df = context.data.copy()

        try:
            # Perform groupby aggregation
            agg_df = df.groupby(self.group_by).agg(self.aggregations)

            # Flatten multi-level columns if needed
            if isinstance(agg_df.columns, pd.MultiIndex):
                agg_df.columns = ['_'.join(col).strip()
                                  for col in agg_df.columns.values]

            agg_df = agg_df.reset_index()

            logger.info(f"Aggregated data: {len(df)} -> {len(agg_df)} rows")

            context.data = agg_df
            context.metadata['aggregator'] = {
                'group_by': self.group_by,
                'input_rows': len(df),
                'output_rows': len(agg_df)
            }

        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            raise

        return self._handle_next(context)


class PivotAggregator(DataProcessor):
    """Create pivot tables"""

    def __init__(self, index: List[str], columns: str, values: str,
                 aggfunc: str = 'mean', fill_value: Any = None):
        """
        Initialize pivot aggregator

        Args:
            index: Columns to use as index
            columns: Column to pivot
            values: Column with values to aggregate
            aggfunc: Aggregation function
            fill_value: Value to fill missing cells
        """
        super().__init__()
        self.index = index
        self.columns = columns
        self.values = values
        self.aggfunc = aggfunc
        self.fill_value = fill_value

    def process(self, context: DataContext) -> DataContext:
        """Create pivot table"""
        df = context.data.copy()

        try:
            pivot_df = df.pivot_table(
                index=self.index,
                columns=self.columns,
                values=self.values,
                aggfunc=self.aggfunc,
                fill_value=self.fill_value
            )

            pivot_df = pivot_df.reset_index()

            logger.info(f"Created pivot table: {pivot_df.shape}")

            context.data = pivot_df
            context.metadata['pivot_aggregator'] = {
                'index': self.index,
                'columns': self.columns,
                'values': self.values,
                'aggfunc': self.aggfunc
            }

        except Exception as e:
            logger.error(f"Pivot aggregation failed: {e}")
            raise

        return self._handle_next(context)


class TimeSeriesAggregator(DataProcessor):
    """Aggregate time series data"""

    def __init__(self, datetime_column: str, frequency: str,
                 aggregations: Dict[str, str]):
        """
        Initialize time series aggregator

        Args:
            datetime_column: Column with datetime values
            frequency: Resampling frequency ('D', 'W', 'M', 'Q', 'Y')
            aggregations: Dict mapping columns to aggregation functions
        """
        super().__init__()
        self.datetime_column = datetime_column
        self.frequency = frequency
        self.aggregations = aggregations

    def process(self, context: DataContext) -> DataContext:
        """Aggregate time series"""
        df = context.data.copy()

        try:
            # Convert to datetime
            df[self.datetime_column] = pd.to_datetime(df[self.datetime_column])

            # Set as index and resample
            df = df.set_index(self.datetime_column)
            agg_df = df.resample(self.frequency).agg(self.aggregations)
            agg_df = agg_df.reset_index()

            logger.info(f"Time series aggregated: {self.frequency}")

            context.data = agg_df
            context.metadata['timeseries_aggregator'] = {
                'datetime_column': self.datetime_column,
                'frequency': self.frequency,
                'periods': len(agg_df)
            }

        except Exception as e:
            logger.error(f"Time series aggregation failed: {e}")
            raise

        return self._handle_next(context)


class RollingAggregator(DataProcessor):
    """Apply rolling window aggregations"""

    def __init__(self, window: int, aggregations: Dict[str, str],
                 min_periods: Optional[int] = None):
        """
        Initialize rolling aggregator

        Args:
            window: Size of rolling window
            aggregations: Dict mapping columns to functions
            min_periods: Minimum observations in window
        """
        super().__init__()
        self.window = window
        self.aggregations = aggregations
        self.min_periods = min_periods or 1

    def process(self, context: DataContext) -> DataContext:
        """Apply rolling aggregations"""
        df = context.data.copy()

        try:
            for col, func in self.aggregations.items():
                if col in df.columns:
                    new_col_name = f'{col}_rolling_{func}_{self.window}'
                    df[new_col_name] = df[col].rolling(
                        window=self.window,
                        min_periods=self.min_periods
                    ).agg(func)

            logger.info(f"Applied rolling aggregation: window={self.window}")

            context.data = df
            context.metadata['rolling_aggregator'] = {
                'window': self.window,
                'aggregations': self.aggregations
            }

        except Exception as e:
            logger.error(f"Rolling aggregation failed: {e}")
            raise

        return self._handle_next(context)


class CustomAggregator(DataProcessor):
    """Apply custom aggregation functions"""

    def __init__(self, group_by: List[str], custom_funcs: Dict[str, Callable]):
        """
        Initialize custom aggregator

        Args:
            group_by: Columns to group by
            custom_funcs: Dict mapping columns to custom functions
        """
        super().__init__()
        self.group_by = group_by
        self.custom_funcs = custom_funcs

    def process(self, context: DataContext) -> DataContext:
        """Apply custom aggregations"""
        df = context.data.copy()

        try:
            agg_df = df.groupby(self.group_by).agg(self.custom_funcs)
            agg_df = agg_df.reset_index()

            logger.info(f"Applied custom aggregations")

            context.data = agg_df
            context.metadata['custom_aggregator'] = {
                'group_by': self.group_by,
                'functions': list(self.custom_funcs.keys())
            }

        except Exception as e:
            logger.error(f"Custom aggregation failed: {e}")
            raise

        return self._handle_next(context)


class CategoricalAggregator(DataProcessor):
    """Aggregate categorical data"""

    def __init__(self, group_by: List[str], categorical_cols: List[str],
                 method: str = 'mode'):
        """
        Initialize categorical aggregator

        Args:
            group_by: Columns to group by
            categorical_cols: Categorical columns to aggregate
            method: Aggregation method ('mode', 'first', 'last', 'count_unique')
        """
        super().__init__()
        self.group_by = group_by
        self.categorical_cols = categorical_cols
        self.method = method

    def process(self, context: DataContext) -> DataContext:
        """Aggregate categorical data"""
        df = context.data.copy()

        try:
            agg_dict = {}

            for col in self.categorical_cols:
                if self.method == 'mode':
                    agg_dict[col] = lambda x: x.mode()[0] if len(
                        x.mode()) > 0 else None
                elif self.method == 'first':
                    agg_dict[col] = 'first'
                elif self.method == 'last':
                    agg_dict[col] = 'last'
                elif self.method == 'count_unique':
                    agg_dict[col] = 'nunique'

            agg_df = df.groupby(self.group_by).agg(agg_dict)
            agg_df = agg_df.reset_index()

            logger.info(f"Aggregated categorical data using {self.method}")

            context.data = agg_df
            context.metadata['categorical_aggregator'] = {
                'method': self.method,
                'columns': self.categorical_cols
            }

        except Exception as e:
            logger.error(f"Categorical aggregation failed: {e}")
            raise

        return self._handle_next(context)
