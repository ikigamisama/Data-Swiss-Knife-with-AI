from abc import ABC, abstractmethod, ABCMeta
from typing import Any, Dict, List, Optional
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class DataFormat(Enum):
    """Supported data formats"""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PARQUET = "parquet"
    SQL = "sql"
    API = "api"


@dataclass
class DataContext:
    """Context object for passing data through pipeline"""
    data: pd.DataFrame
    metadata: Dict[str, Any]
    source: str
    timestamp: str


# ============= Strategy Pattern: Data Loaders =============
class DataLoader(ABC):
    """Abstract base class for data loading strategies"""

    @abstractmethod
    def load(self, source: Any, **kwargs) -> pd.DataFrame:
        """Load data from source"""
        pass

    @abstractmethod
    def validate(self, source: Any) -> bool:
        """Validate if source is compatible with this loader"""
        pass


# ============= Chain of Responsibility: Data Processing =============
class DataProcessor(ABC):
    """Abstract processor for chain of responsibility pattern"""

    def __init__(self):
        self._next_processor: Optional[DataProcessor] = None

    def set_next(self, processor: 'DataProcessor') -> 'DataProcessor':
        """Set the next processor in chain"""
        self._next_processor = processor
        return processor

    @abstractmethod
    def process(self, context: DataContext) -> DataContext:
        """Process data and pass to next processor"""
        pass

    def _handle_next(self, context: DataContext) -> DataContext:
        """Pass to next processor if exists"""
        if self._next_processor:
            return self._next_processor.process(context)
        return context


# ============= Factory Pattern: Visualizations =============
class Visualization(ABC):
    """Abstract base for visualization factory"""

    @abstractmethod
    def create(self, data: pd.DataFrame, **kwargs) -> Any:
        """Create visualization"""
        pass

    @abstractmethod
    def get_type(self) -> str:
        """Return visualization type"""
        pass


class VisualizationFactory(ABC):
    """Factory for creating visualizations"""

    @abstractmethod
    def create_visualization(self, viz_type: str) -> Visualization:
        """Create visualization based on type"""
        pass


# ============= Observer Pattern: Data Quality Monitoring =============
class DataQualityObserver(ABC):
    """Observer for data quality events"""

    @abstractmethod
    def update(self, event: str, data: Dict[str, Any]) -> None:
        """Receive quality check updates"""
        pass


class DataQualitySubject:
    """Subject that notifies observers of quality issues"""

    def __init__(self):
        self._observers: List[DataQualityObserver] = []

    def attach(self, observer: DataQualityObserver) -> None:
        """Attach an observer"""
        self._observers.append(observer)

    def detach(self, observer: DataQualityObserver) -> None:
        """Detach an observer"""
        self._observers.remove(observer)

    def notify(self, event: str, data: Dict[str, Any]) -> None:
        """Notify all observers"""
        for observer in self._observers:
            observer.update(event, data)


# ============= Template Method: Analysis Pipeline =============
class AnalysisPipeline(ABC):
    """Template method for analysis pipelines"""

    def execute(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Template method defining the analysis workflow"""
        self._validate_data(data)
        prepared_data = self._prepare_data(data)
        results = self._analyze(prepared_data)
        visualizations = self._visualize(results, prepared_data)
        report = self._generate_report(results, visualizations)
        return report

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data"""
        if data.empty:
            raise ValueError("Empty dataframe provided")

    @abstractmethod
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for analysis"""
        pass

    @abstractmethod
    def _analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform analysis"""
        pass

    @abstractmethod
    def _visualize(self, results: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Create visualizations"""
        pass

    @abstractmethod
    def _generate_report(self, results: Dict[str, Any], viz: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final report"""
        pass


# ============= Singleton: Configuration Manager =============
class SingletonMeta(ABCMeta):
    """Metaclass for Singleton pattern"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# ============= Builder Pattern: Query Builder =============
class QueryBuilder(ABC):
    """Abstract builder for constructing queries"""

    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Reset builder to initial state"""
        pass

    @abstractmethod
    def select(self, columns: List[str]) -> 'QueryBuilder':
        """Add SELECT clause"""
        pass

    @abstractmethod
    def where(self, condition: str) -> 'QueryBuilder':
        """Add WHERE clause"""
        pass

    @abstractmethod
    def group_by(self, columns: List[str]) -> 'QueryBuilder':
        """Add GROUP BY clause"""
        pass

    @abstractmethod
    def order_by(self, column: str, ascending: bool = True) -> 'QueryBuilder':
        """Add ORDER BY clause"""
        pass

    @abstractmethod
    def build(self) -> Any:
        """Build and return the query"""
        pass


# ============= Adapter Pattern: LLM Integration =============
class LLMAdapter(ABC):
    """Adapter for different LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM"""
        pass

    @abstractmethod
    def generate_code(self, instruction: str, context: str) -> str:
        """Generate code from instruction"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        pass
