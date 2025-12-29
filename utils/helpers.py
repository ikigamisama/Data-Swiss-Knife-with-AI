import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats
import re


def format_number(num: int) -> str:
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect column types in a DataFrame
    Returns dict with keys: numeric, categorical, datetime, text, boolean
    """
    types = {
        'numeric': [],
        'categorical': [],
        'datetime': [],
        'text': [],
        'boolean': []
    }

    for col in df.columns:
        dtype = df[col].dtype

        # Boolean
        if dtype == bool or df[col].nunique() == 2:
            types['boolean'].append(col)

        # Numeric
        elif np.issubdtype(dtype, np.number):
            types['numeric'].append(col)

        # Datetime
        elif np.issubdtype(dtype, np.datetime64):
            types['datetime'].append(col)

        # Try to infer datetime from string
        elif dtype == object:
            sample = df[col].dropna().head(100)

            # Try datetime parsing
            try:
                parsed = pd.to_datetime(sample, errors='coerce')
                if parsed.notna().sum() / len(sample) > 0.8:
                    types['datetime'].append(col)
                    continue
            except:
                pass

            # Categorical or text
            unique_ratio = df[col].nunique() / len(df)
            avg_length = df[col].astype(str).str.len().mean()

            if unique_ratio < 0.5 and df[col].nunique() < 50:
                types['categorical'].append(col)
            else:
                types['text'].append(col)

    return types


def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data quality report
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicates': df.duplicated().sum(),
        'missing_values': {},
        'column_stats': {}
    }

    # Missing values analysis
    missing = df.isnull().sum()
    report['missing_values'] = {
        'total': missing.sum(),
        'by_column': missing[missing > 0].to_dict(),
        'percentage': (missing.sum() / (len(df) * len(df.columns)) * 100)
    }

    # Column statistics
    for col in df.columns:
        stats = {
            'dtype': str(df[col].dtype),
            'non_null': df[col].count(),
            'null': df[col].isnull().sum(),
            'unique': df[col].nunique(),
            'unique_ratio': df[col].nunique() / len(df)
        }

        if np.issubdtype(df[col].dtype, np.number):
            stats.update({
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'skew': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            })

            # Outlier detection using IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) |
                        (df[col] > (Q3 + 1.5 * IQR))).sum()
            stats['outliers'] = outliers

        report['column_stats'][col] = stats

    return report


def suggest_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Suggest optimal data types for columns
    """
    suggestions = {}

    for col in df.columns:
        current_type = df[col].dtype

        # Check if object column can be converted to numeric
        if current_type == object:
            try:
                pd.to_numeric(df[col], errors='raise')
                suggestions[col] = 'numeric (int or float)'
            except:
                pass

            # Check if it's a date
            try:
                pd.to_datetime(df[col], errors='raise')
                suggestions[col] = 'datetime'
            except:
                pass

            # Check if it should be categorical
            if df[col].nunique() / len(df) < 0.5:
                suggestions[col] = 'category'

        # Check if numeric can be downcasted
        elif np.issubdtype(current_type, np.integer):
            min_val = df[col].min()
            max_val = df[col].max()

            if min_val >= 0:
                if max_val <= 255:
                    suggestions[col] = 'uint8'
                elif max_val <= 65535:
                    suggestions[col] = 'uint16'
            else:
                if min_val >= -128 and max_val <= 127:
                    suggestions[col] = 'int8'
                elif min_val >= -32768 and max_val <= 32767:
                    suggestions[col] = 'int16'

    return suggestions


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names: lowercase, remove special chars, replace spaces
    """
    df = df.copy()

    new_columns = []
    for col in df.columns:
        # Convert to string and lowercase
        new_col = str(col).lower()

        # Remove special characters except underscore
        new_col = re.sub(r'[^a-z0-9_]', '_', new_col)

        # Remove multiple underscores
        new_col = re.sub(r'_+', '_', new_col)

        # Remove leading/trailing underscores
        new_col = new_col.strip('_')

        new_columns.append(new_col)

    df.columns = new_columns
    return df


def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in numeric columns

    Args:
        df: DataFrame
        method: 'iqr', 'zscore', or 'isolation_forest'
        threshold: Threshold for outlier detection

    Returns:
        DataFrame with boolean columns indicating outliers
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_df = pd.DataFrame(index=df.index)

    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_df[f'{col}_outlier'] = (
                df[col] < lower_bound) | (df[col] > upper_bound)

        elif method == 'zscore':

            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_df[f'{col}_outlier'] = False
            outlier_df.loc[df[col].notna(
            ), f'{col}_outlier'] = z_scores > threshold

    return outlier_df


def generate_sample_data(n_rows: int = 1000, scenario: str = 'sales') -> pd.DataFrame:
    """
    Generate sample data for testing

    Args:
        n_rows: Number of rows
        scenario: 'sales', 'hr', 'iot', 'finance'
    """
    np.random.seed(42)

    if scenario == 'sales':
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n_rows, freq='D'),
            'product': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
            'sales': np.random.gamma(2, 1000, n_rows),
            'quantity': np.random.poisson(10, n_rows),
            'discount': np.random.uniform(0, 0.3, n_rows),
            'profit': np.random.normal(500, 200, n_rows)
        })

    elif scenario == 'hr':
        return pd.DataFrame({
            'employee_id': range(1, n_rows + 1),
            'name': [f'Employee_{i}' for i in range(1, n_rows + 1)],
            'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing', 'Finance'], n_rows),
            'salary': np.random.normal(70000, 20000, n_rows),
            'experience_years': np.random.randint(0, 30, n_rows),
            'performance_score': np.random.uniform(1, 5, n_rows),
            'hire_date': pd.date_range('2010-01-01', periods=n_rows, freq='W')
        })

    elif scenario == 'iot':
        return pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n_rows, freq='min'),
            'device_id': np.random.choice([f'DEV_{i}' for i in range(1, 11)], n_rows),
            'temperature': np.random.normal(25, 5, n_rows),
            'humidity': np.random.normal(60, 15, n_rows),
            'pressure': np.random.normal(1013, 10, n_rows),
            'battery': np.random.uniform(0, 100, n_rows),
            'status': np.random.choice(['active', 'idle', 'error'], n_rows, p=[0.7, 0.2, 0.1])
        })

    elif scenario == 'finance':
        dates = pd.date_range('2023-01-01', periods=n_rows, freq='D')
        price = 100
        prices = [price]
        for _ in range(n_rows - 1):
            price = price * (1 + np.random.normal(0.001, 0.02))
            prices.append(price)

        return pd.DataFrame({
            'date': dates,
            'open': prices,
            'close': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
            'volume': np.random.lognormal(15, 1, n_rows)
        })

    return pd.DataFrame()


def export_to_multiple_formats(df: pd.DataFrame, base_filename: str) -> Dict[str, str]:
    exports = {}

    # CSV
    csv_path = f"{base_filename}.csv"
    df.to_csv(csv_path, index=False)
    exports['csv'] = csv_path

    # Excel
    excel_path = f"{base_filename}.xlsx"
    df.to_excel(excel_path, index=False)
    exports['excel'] = excel_path

    # JSON
    json_path = f"{base_filename}.json"
    df.to_json(json_path, orient='records', indent=2)
    exports['json'] = json_path

    # Parquet
    parquet_path = f"{base_filename}.parquet"
    df.to_parquet(parquet_path, index=False)
    exports['parquet'] = parquet_path

    return exports


def make_pyarrow_friendly(df):
    df = df.copy()   # 👈 critical line

    for col in df.columns:
        if str(df[col].dtype).startswith("Int"):
            df[col] = df[col].astype("float64")

        if df[col].dtype == "object":
            df[col] = df[col].astype(str)

    return df