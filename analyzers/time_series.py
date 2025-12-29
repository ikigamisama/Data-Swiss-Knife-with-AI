"""
Time Series Analyzer - Comprehensive Time Series Analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """Analyze time series data"""
    
    def __init__(self, df: pd.DataFrame, date_column: str, value_column: str):
        """
        Initialize time series analyzer
        
        Args:
            df: DataFrame with time series data
            date_column: Column with datetime values
            value_column: Column with values to analyze
        """
        self.df = df.copy()
        self.date_column = date_column
        self.value_column = value_column
        
        # Convert to datetime and sort
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        self.df = self.df.sort_values(date_column)
        self.df = self.df.set_index(date_column)
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform comprehensive time series analysis
        
        Returns:
            Dictionary with analysis results
        """
        try:
            results = {
                'basic_stats': self._get_basic_stats(),
                'trend': self._analyze_trend(),
                'seasonality': self._detect_seasonality(),
                'stationarity': self._test_stationarity(),
                'autocorrelation': self._analyze_autocorrelation(),
                'forecast': self._simple_forecast()
            }
            
            logger.info("Time series analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"Time series analysis failed: {e}")
            return {'error': str(e)}
    
    def _get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics"""
        series = self.df[self.value_column]
        
        return {
            'count': int(len(series)),
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'start_date': str(series.index[0]),
            'end_date': str(series.index[-1]),
            'duration_days': (series.index[-1] - series.index[0]).days
        }
    
    def _analyze_trend(self) -> Dict[str, Any]:
        """Analyze trend using linear regression"""
        from scipy import stats
        
        series = self.df[self.value_column].dropna()
        x = np.arange(len(series))
        y = series.values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine trend type
        if p_value < 0.05:
            if slope > 0:
                trend_type = 'increasing'
            else:
                trend_type = 'decreasing'
        else:
            trend_type = 'no_significant_trend'
        
        return {
            'type': trend_type,
            'slope': float(slope),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
    
    def _detect_seasonality(self) -> Dict[str, Any]:
        """Detect seasonality patterns"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            series = self.df[self.value_column].dropna()
            
            # Determine period
            freq = pd.infer_freq(series.index)
            
            if freq is None:
                # Resample to regular frequency
                series = series.resample('D').mean().dropna()
            
            if len(series) < 24:
                return {'detected': False, 'reason': 'Insufficient data'}
            
            # Perform decomposition
            period = min(12, len(series) // 2)
            decomposition = seasonal_decompose(
                series, 
                model='additive', 
                period=period,
                extrapolate_trend='freq'
            )
            
            # Calculate seasonality strength
            seasonal_strength = (
                decomposition.seasonal.var() / 
                (decomposition.seasonal.var() + decomposition.resid.var())
            )
            
            return {
                'detected': seasonal_strength > 0.3,
                'strength': float(seasonal_strength),
                'period': period,
                'seasonal_component_available': True
            }
            
        except Exception as e:
            logger.warning(f"Seasonality detection failed: {e}")
            return {'detected': False, 'reason': str(e)}
    
    def _test_stationarity(self) -> Dict[str, Any]:
        """Test for stationarity using Augmented Dickey-Fuller test"""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            series = self.df[self.value_column].dropna()
            
            result = adfuller(series, autolag='AIC')
            
            return {
                'is_stationary': result[1] < 0.05,
                'adf_statistic': float(result[0]),
                'p_value': float(result[1]),
                'critical_values': {k: float(v) for k, v in result[4].items()}
            }
            
        except Exception as e:
            logger.warning(f"Stationarity test failed: {e}")
            return {'error': str(e)}
    
    def _analyze_autocorrelation(self) -> Dict[str, Any]:
        """Analyze autocorrelation"""
        try:
            from statsmodels.tsa.stattools import acf, pacf
            
            series = self.df[self.value_column].dropna()
            
            # Calculate ACF and PACF
            nlags = min(40, len(series) // 2)
            acf_values = acf(series, nlags=nlags)
            pacf_values = pacf(series, nlags=nlags)
            
            # Find significant lags
            significant_lags = []
            confidence_interval = 1.96 / np.sqrt(len(series))
            
            for lag in range(1, len(acf_values)):
                if abs(acf_values[lag]) > confidence_interval:
                    significant_lags.append(lag)
            
            return {
                'significant_lags': significant_lags[:10],
                'max_autocorrelation': float(max(abs(acf_values[1:]))),
                'has_autocorrelation': len(significant_lags) > 0
            }
            
        except Exception as e:
            logger.warning(f"Autocorrelation analysis failed: {e}")
            return {'error': str(e)}
    
    def _simple_forecast(self, periods: int = 7) -> Dict[str, Any]:
        """Simple forecast using exponential smoothing"""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            series = self.df[self.value_column].dropna()
            
            # Fit exponential smoothing
            model = ExponentialSmoothing(
                series,
                seasonal_periods=7,
                trend='add',
                seasonal='add'
            )
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(periods)
            
            return {
                'method': 'exponential_smoothing',
                'periods': periods,
                'forecast_values': forecast.tolist(),
                'confidence_available': False
            }
            
        except Exception as e:
            # Fallback to simple mean forecast
            logger.warning(f"Exponential smoothing failed, using mean: {e}")
            mean_value = self.df[self.value_column].mean()
            
            return {
                'method': 'mean',
                'periods': periods,
                'forecast_values': [mean_value] * periods,
                'confidence_available': False
            }
    
    def detect_anomalies(self, window: int = 7, threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect anomalies in time series
        
        Args:
            window: Rolling window size
            threshold: Z-score threshold
            
        Returns:
            Dictionary with anomaly information
        """
        try:
            series = self.df[self.value_column]
            
            # Calculate rolling statistics
            rolling_mean = series.rolling(window=window, center=True).mean()
            rolling_std = series.rolling(window=window, center=True).std()
            
            # Calculate z-scores
            z_scores = np.abs((series - rolling_mean) / rolling_std)
            
            # Find anomalies
            anomalies = series[z_scores > threshold]
            
            return {
                'total_anomalies': len(anomalies),
                'anomaly_percentage': len(anomalies) / len(series) * 100,
                'anomaly_dates': [str(d) for d in anomalies.index[:20]],
                'anomaly_values': anomalies.values[:20].tolist()
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {'error': str(e)}
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate time series metrics"""
        series = self.df[self.value_column]
        
        # Growth rate
        first_value = series.iloc[0]
        last_value = series.iloc[-1]
        total_growth = (last_value - first_value) / first_value * 100
        
        # Volatility
        returns = series.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Moving averages
        ma_7 = series.rolling(window=7).mean().iloc[-1]
        ma_30 = series.rolling(window=30).mean().iloc[-1]
        
        return {
            'total_growth_pct': float(total_growth),
            'volatility': float(volatility),
            'current_value': float(last_value),
            'ma_7': float(ma_7) if not np.isnan(ma_7) else None,
            'ma_30': float(ma_30) if not np.isnan(ma_30) else None,
            'max_drawdown': float(self._calculate_max_drawdown(series))
        }
    
    def _calculate_max_drawdown(self, series: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cummax = series.cummax()
        drawdown = (series - cummax) / cummax
        return drawdown.min()