"""
Custom Exceptions for Data Analysis Swiss Knife
"""


class DataAnalysisError(Exception):
    """Base exception for all data analysis errors"""
    pass


class DataLoadError(DataAnalysisError):
    """Exception raised when data loading fails"""
    pass


class DataValidationError(DataAnalysisError):
    """Exception raised when data validation fails"""
    pass


class DataProcessingError(DataAnalysisError):
    """Exception raised during data processing"""
    pass


class DataQualityError(DataAnalysisError):
    """Exception raised when data quality checks fail"""
    pass


class LLMError(DataAnalysisError):
    """Base exception for LLM-related errors"""
    pass


class LLMConnectionError(LLMError):
    """Exception raised when LLM connection fails"""
    pass


class LLMTimeoutError(LLMError):
    """Exception raised when LLM request times out"""
    pass


class LLMResponseError(LLMError):
    """Exception raised when LLM response is invalid"""
    pass


class DatabaseError(DataAnalysisError):
    """Base exception for database errors"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Exception raised when database connection fails"""
    pass


class QueryExecutionError(DatabaseError):
    """Exception raised when query execution fails"""
    pass


class PipelineError(DataAnalysisError):
    """Exception raised during pipeline execution"""
    pass


class ConfigurationError(DataAnalysisError):
    """Exception raised for configuration issues"""
    pass


class VisualizationError(DataAnalysisError):
    """Exception raised during visualization creation"""
    pass


class ModelTrainingError(DataAnalysisError):
    """Exception raised during model training"""
    pass


class FeatureEngineeringError(DataAnalysisError):
    """Exception raised during feature engineering"""
    pass
