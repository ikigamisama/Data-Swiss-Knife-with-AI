import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Perform various statistical analyses"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize statistical analyzer

        Args:
            df: DataFrame to analyze
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(
            include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(
            include=['object', 'category']).columns.tolist()

    def descriptive_statistics(self) -> Dict[str, Any]:
        """Calculate descriptive statistics"""
        try:
            results = {
                'numeric': {},
                'categorical': {}
            }

            # Numeric statistics
            for col in self.numeric_cols:
                results['numeric'][col] = {
                    'count': int(self.df[col].count()),
                    'mean': float(self.df[col].mean()),
                    'median': float(self.df[col].median()),
                    'std': float(self.df[col].std()),
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max()),
                    'q25': float(self.df[col].quantile(0.25)),
                    'q75': float(self.df[col].quantile(0.75)),
                    'skewness': float(self.df[col].skew()),
                    'kurtosis': float(self.df[col].kurtosis())
                }

            # Categorical statistics
            for col in self.categorical_cols:
                value_counts = self.df[col].value_counts().to_dict()
                results['categorical'][col] = {
                    'unique': int(self.df[col].nunique()),
                    'mode': str(self.df[col].mode()[0]) if len(self.df[col].mode()) > 0 else None,
                    'top_values': {str(k): int(v) for k, v in list(value_counts.items())[:5]}
                }

            logger.info("Descriptive statistics calculated")
            return results

        except Exception as e:
            logger.error(f"Error calculating descriptive statistics: {e}")
            raise

    def correlation_analysis(self, method: str = 'pearson') -> Dict[str, Any]:
        """Calculate correlation matrix"""
        try:
            if len(self.numeric_cols) < 2:
                return {'error': 'Need at least 2 numeric columns'}

            corr_matrix = self.df[self.numeric_cols].corr(method=method)

            # Find strong correlations
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': float(corr_val)
                        })

            results = {
                'method': method,
                'matrix': corr_matrix.to_dict(),
                'strong_correlations': strong_corr
            }

            logger.info(f"Correlation analysis completed using {method}")
            return results

        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            raise

    def normality_test(self, alpha: float = 0.05) -> Dict[str, Any]:
        """Test normality using Shapiro-Wilk test"""
        try:
            results = {}

            for col in self.numeric_cols:
                # Sample if data is too large
                sample = self.df[col].dropna()
                if len(sample) > 5000:
                    sample = sample.sample(5000)

                statistic, p_value = stats.shapiro(sample)

                results[col] = {
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'is_normal': p_value > alpha,
                    'alpha': alpha
                }

            logger.info("Normality tests completed")
            return results

        except Exception as e:
            logger.error(f"Error in normality test: {e}")
            raise

    def outlier_detection(self, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, Any]:
        """Detect outliers in numeric columns"""
        try:
            results = {}

            for col in self.numeric_cols:
                data = self.df[col].dropna()

                if method == 'iqr':
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outliers = ((data < lower_bound) |
                                (data > upper_bound)).sum()

                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(data))
                    outliers = (z_scores > threshold).sum()

                else:
                    raise ValueError(f"Unknown method: {method}")

                results[col] = {
                    'count': int(outliers),
                    'percentage': float(outliers / len(data) * 100),
                    'method': method
                }

            logger.info(f"Outlier detection completed using {method}")
            return results

        except Exception as e:
            logger.error(f"Error in outlier detection: {e}")
            raise

    def hypothesis_test(self, test_type: str, **kwargs) -> Dict[str, Any]:
        """Perform hypothesis tests"""
        try:
            if test_type == 't_test':
                group_col = kwargs['group_col']
                value_col = kwargs['value_col']

                groups = self.df[group_col].unique()
                if len(groups) < 2:
                    return {'error': 'Need at least 2 groups'}

                group1 = self.df[self.df[group_col]
                                 == groups[0]][value_col].dropna()
                group2 = self.df[self.df[group_col]
                                 == groups[1]][value_col].dropna()

                statistic, p_value = stats.ttest_ind(group1, group2)

                return {
                    'test': 't_test',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'group1_mean': float(group1.mean()),
                    'group2_mean': float(group2.mean())
                }

            elif test_type == 'chi_square':
                col1 = kwargs['col1']
                col2 = kwargs['col2']

                contingency_table = pd.crosstab(self.df[col1], self.df[col2])
                chi2, p_value, dof, expected = stats.chi2_contingency(
                    contingency_table)

                return {
                    'test': 'chi_square',
                    'chi2': float(chi2),
                    'p_value': float(p_value),
                    'dof': int(dof),
                    'significant': p_value < 0.05
                }

            else:
                raise ValueError(f"Unknown test type: {test_type}")

        except Exception as e:
            logger.error(f"Error in hypothesis test: {e}")
            raise

    def distribution_analysis(self, column: str) -> Dict[str, Any]:
        """Analyze distribution of a column"""
        try:
            data = self.df[column].dropna()

            # Basic statistics
            results = {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'mode': float(data.mode()[0]) if len(data.mode()) > 0 else None,
                'std': float(data.std()),
                'variance': float(data.var()),
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis()),
                'min': float(data.min()),
                'max': float(data.max())
            }

            # Test for different distributions
            distributions = {}

            # Normal distribution
            _, p_normal = stats.shapiro(data.sample(min(5000, len(data))))
            distributions['normal'] = {'p_value': float(
                p_normal), 'fits': p_normal > 0.05}

            # Exponential distribution
            _, p_exp = stats.kstest(data, 'expon')
            distributions['exponential'] = {
                'p_value': float(p_exp), 'fits': p_exp > 0.05}

            results['distribution_tests'] = distributions

            logger.info(f"Distribution analysis completed for {column}")
            return results

        except Exception as e:
            logger.error(f"Error in distribution analysis: {e}")
            raise

    def trend_analysis(self, column: str) -> Dict[str, Any]:
        """Analyze trend in time series or sequential data"""
        try:
            data = self.df[column].dropna()
            x = np.arange(len(data))
            y = data.values

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x, y)

            # Determine trend direction
            if p_value < 0.05:
                if slope > 0:
                    trend = 'increasing'
                else:
                    trend = 'decreasing'
            else:
                trend = 'no_significant_trend'

            results = {
                'trend': trend,
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }

            logger.info(f"Trend analysis completed for {column}")
            return results

        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            raise

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive statistical report"""
        try:
            report = {
                'dataset_info': {
                    'rows': len(self.df),
                    'columns': len(self.df.columns),
                    'numeric_columns': len(self.numeric_cols),
                    'categorical_columns': len(self.categorical_cols),
                    'missing_values': int(self.df.isnull().sum().sum())
                },
                'descriptive_statistics': self.descriptive_statistics(),
                'correlation_analysis': self.correlation_analysis() if len(self.numeric_cols) >= 2 else None,
                'normality_tests': self.normality_test(),
                'outlier_detection': self.outlier_detection()
            }

            logger.info("Statistical report generated")
            return report

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
