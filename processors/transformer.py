import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from core.base import DataProcessor, DataContext
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer(DataProcessor):
    """Feature engineering transformations"""

    def __init__(self, operations: List[Dict[str, Any]]):
        """
        Initialize feature engineer

        Args:
            operations: List of feature engineering operations
                Example: [
                    {'type': 'polynomial', 'columns': ['age'], 'degree': 2},
                    {'type': 'interaction', 'columns': ['col1', 'col2']},
                    {'type': 'binning', 'column': 'age', 'bins': [0, 18, 65, 100]}
                ]
        """
        super().__init__()
        self.operations = operations

    def process(self, context: DataContext) -> DataContext:
        """Apply feature engineering operations"""
        df = context.data.copy()

        try:
            for op in self.operations:
                op_type = op.get('type')

                if op_type == 'polynomial':
                    df = self._create_polynomial_features(df, op)
                elif op_type == 'interaction':
                    df = self._create_interaction_features(df, op)
                elif op_type == 'binning':
                    df = self._create_binned_features(df, op)
                elif op_type == 'log':
                    df = self._create_log_features(df, op)
                elif op_type == 'sqrt':
                    df = self._create_sqrt_features(df, op)
                elif op_type == 'datetime':
                    df = self._extract_datetime_features(df, op)
                elif op_type == 'aggregation':
                    df = self._create_aggregation_features(df, op)

                logger.info(f"Applied feature engineering: {op_type}")

            context.data = df
            context.metadata['feature_engineer'] = {
                'operations': len(self.operations),
                'new_columns': len(df.columns) - len(context.data.columns)
            }

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise

        return self._handle_next(context)

    def _create_polynomial_features(self, df: pd.DataFrame, op: Dict) -> pd.DataFrame:
        """Create polynomial features"""
        columns = op['columns']
        degree = op.get('degree', 2)

        for col in columns:
            for d in range(2, degree + 1):
                df[f'{col}_pow{d}'] = df[col] ** d

        return df

    def _create_interaction_features(self, df: pd.DataFrame, op: Dict) -> pd.DataFrame:
        """Create interaction features"""
        columns = op['columns']

        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]

        return df

    def _create_binned_features(self, df: pd.DataFrame, op: Dict) -> pd.DataFrame:
        """Create binned features"""
        column = op['column']
        bins = op['bins']
        labels = op.get('labels', None)

        df[f'{column}_binned'] = pd.cut(df[column], bins=bins, labels=labels)

        return df

    def _create_log_features(self, df: pd.DataFrame, op: Dict) -> pd.DataFrame:
        """Create log-transformed features"""
        columns = op['columns']

        for col in columns:
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))

        return df

    def _create_sqrt_features(self, df: pd.DataFrame, op: Dict) -> pd.DataFrame:
        """Create square root features"""
        columns = op['columns']

        for col in columns:
            df[f'{col}_sqrt'] = np.sqrt(df[col].clip(lower=0))

        return df

    def _extract_datetime_features(self, df: pd.DataFrame, op: Dict) -> pd.DataFrame:
        """Extract datetime features"""
        column = op['column']

        # Convert to datetime
        df[column] = pd.to_datetime(df[column])

        # Extract features
        df[f'{column}_year'] = df[column].dt.year
        df[f'{column}_month'] = df[column].dt.month
        df[f'{column}_day'] = df[column].dt.day
        df[f'{column}_dayofweek'] = df[column].dt.dayofweek
        df[f'{column}_quarter'] = df[column].dt.quarter
        df[f'{column}_is_weekend'] = df[column].dt.dayofweek.isin(
            [5, 6]).astype(int)

        return df

    def _create_aggregation_features(self, df: pd.DataFrame, op: Dict) -> pd.DataFrame:
        """Create aggregation features"""
        groupby_cols = op['groupby']
        agg_col = op['column']
        agg_funcs = op.get('functions', ['mean', 'sum'])

        for func in agg_funcs:
            agg_values = df.groupby(groupby_cols)[agg_col].transform(func)
            df[f'{agg_col}_{func}_by_{"_".join(groupby_cols)}'] = agg_values

        return df


class ScalingTransformer(DataProcessor):
    """Scale numeric features"""

    def __init__(self, method: str = 'standard', columns: Optional[List[str]] = None):
        """
        Initialize scaling transformer

        Args:
            method: 'standard', 'minmax', 'robust'
            columns: Columns to scale (None for all numeric)
        """
        super().__init__()
        self.method = method
        self.columns = columns
        self.scaler = None

    def process(self, context: DataContext) -> DataContext:
        """Scale features"""
        df = context.data.copy()

        try:
            # Select columns to scale
            if self.columns is None:
                scale_cols = df.select_dtypes(
                    include=[np.number]).columns.tolist()
            else:
                scale_cols = self.columns

            # Initialize scaler
            if self.method == 'standard':
                self.scaler = StandardScaler()
            elif self.method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.method}")

            # Fit and transform
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])

            logger.info(
                f"Scaled {len(scale_cols)} columns using {self.method}")

            context.data = df
            context.metadata['scaling_transformer'] = {
                'method': self.method,
                'columns': scale_cols
            }

        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            raise

        return self._handle_next(context)


class EncodingTransformer(DataProcessor):
    """Encode categorical features"""

    def __init__(self, method: str = 'onehot', columns: Optional[List[str]] = None):
        """
        Initialize encoding transformer

        Args:
            method: 'onehot', 'label', 'target'
            columns: Columns to encode (None for all categorical)
        """
        super().__init__()
        self.method = method
        self.columns = columns
        self.encoders = {}

    def process(self, context: DataContext) -> DataContext:
        """Encode categorical features"""
        df = context.data.copy()

        try:
            # Select columns to encode
            if self.columns is None:
                encode_cols = df.select_dtypes(
                    include=['object', 'category']).columns.tolist()
            else:
                encode_cols = self.columns

            if self.method == 'onehot':
                df = pd.get_dummies(df, columns=encode_cols, drop_first=True)

            elif self.method == 'label':
                for col in encode_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le

            logger.info(
                f"Encoded {len(encode_cols)} columns using {self.method}")

            context.data = df
            context.metadata['encoding_transformer'] = {
                'method': self.method,
                'columns': encode_cols
            }

        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise

        return self._handle_next(context)


class AggregationTransformer(DataProcessor):
    """Create aggregated features"""

    def __init__(self, group_by: List[str], aggregations: Dict[str, List[str]]):
        """
        Initialize aggregation transformer

        Args:
            group_by: Columns to group by
            aggregations: Dict mapping columns to aggregation functions
                Example: {'sales': ['sum', 'mean'], 'quantity': ['count']}
        """
        super().__init__()
        self.group_by = group_by
        self.aggregations = aggregations

    def process(self, context: DataContext) -> DataContext:
        """Create aggregated features"""
        df = context.data.copy()

        try:
            # Perform aggregations
            agg_df = df.groupby(self.group_by).agg(self.aggregations)

            # Flatten column names
            agg_df.columns = ['_'.join(col).strip()
                              for col in agg_df.columns.values]
            agg_df = agg_df.reset_index()

            # Merge back to original
            df = df.merge(agg_df, on=self.group_by, how='left')

            logger.info(f"Created {len(agg_df.columns)} aggregated features")

            context.data = df
            context.metadata['aggregation_transformer'] = {
                'group_by': self.group_by,
                'features_created': len(agg_df.columns)
            }

        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            raise

        return self._handle_next(context)
