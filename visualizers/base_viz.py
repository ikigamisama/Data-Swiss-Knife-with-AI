from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseVisualization(ABC):
    """Abstract base class for all visualizations"""

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.figure = None

    @abstractmethod
    def create(self, data: pd.DataFrame, **kwargs) -> Any:
        """
        Create visualization

        Args:
            data: DataFrame to visualize
            **kwargs: Visualization-specific parameters

        Returns:
            Figure object
        """
        pass

    @abstractmethod
    def get_type(self) -> str:
        """
        Get visualization type

        Returns:
            String identifying visualization type
        """
        pass

    def validate_data(self, data: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
        """
        Validate input data

        Args:
            data: DataFrame to validate
            required_columns: List of required columns

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if data.empty:
            raise ValueError("Cannot create visualization with empty data")

        if required_columns:
            missing = [
                col for col in required_columns if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

        return True

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()

    def set_config(self, **kwargs):
        """Update configuration"""
        self.config.update(kwargs)
        return self

    def save(self, filepath: str, **kwargs):
        """
        Save visualization to file

        Args:
            filepath: Path to save file
            **kwargs: Save parameters (format, dpi, etc.)
        """
        if self.figure is None:
            raise ValueError("No figure to save. Call create() first.")

        # Implementation depends on plotting library
        logger.info(f"Saving visualization to {filepath}")


class BaseVisualizationFactory(ABC):
    """Abstract factory for creating visualizations"""

    def __init__(self):
        self._visualizations: Dict[str, BaseVisualization] = {}
        self._register_visualizations()

    @abstractmethod
    def _register_visualizations(self):
        """Register available visualizations"""
        pass

    def register(self, name: str, visualization: BaseVisualization):
        """
        Register a visualization

        Args:
            name: Visualization name
            visualization: Visualization instance
        """
        self._visualizations[name] = visualization
        logger.info(f"Registered visualization: {name}")

    def unregister(self, name: str):
        """Unregister a visualization"""
        if name in self._visualizations:
            del self._visualizations[name]
            logger.info(f"Unregistered visualization: {name}")

    def create_visualization(self, viz_type: str) -> BaseVisualization:
        """
        Create a visualization by type

        Args:
            viz_type: Type of visualization

        Returns:
            Visualization instance

        Raises:
            ValueError: If type not found
        """
        if viz_type not in self._visualizations:
            available = ', '.join(self._visualizations.keys())
            raise ValueError(
                f"Unknown visualization type: {viz_type}. "
                f"Available: {available}"
            )

        return self._visualizations[viz_type]

    def list_available(self) -> List[str]:
        """
        List all available visualization types

        Returns:
            List of visualization names
        """
        return list(self._visualizations.keys())

    def create(self, viz_type: str, data: pd.DataFrame, **kwargs) -> Any:
        """
        Create and return a visualization

        Args:
            viz_type: Type of visualization
            data: Data to visualize
            **kwargs: Visualization parameters

        Returns:
            Figure object
        """
        viz = self.create_visualization(viz_type)
        return viz.create(data, **kwargs)


class VisualizationTheme:
    """Manage visualization themes"""

    THEMES = {
        'default': {
            'color_palette': 'viridis',
            'background': 'white',
            'grid': True,
            'font_size': 12
        },
        'dark': {
            'color_palette': 'plasma',
            'background': '#1e1e1e',
            'grid': True,
            'font_size': 12
        },
        'minimal': {
            'color_palette': 'Blues',
            'background': 'white',
            'grid': False,
            'font_size': 10
        },
        'presentation': {
            'color_palette': 'Set2',
            'background': 'white',
            'grid': True,
            'font_size': 14
        }
    }

    def __init__(self, theme: str = 'default'):
        """
        Initialize theme

        Args:
            theme: Theme name
        """
        self.current_theme = theme
        self.config = self.THEMES.get(theme, self.THEMES['default'])

    def get_config(self) -> Dict[str, Any]:
        """Get theme configuration"""
        return self.config.copy()

    def apply(self, figure: Any) -> Any:
        """
        Apply theme to figure

        Args:
            figure: Figure object

        Returns:
            Themed figure
        """
        # Implementation depends on plotting library
        return figure

    @classmethod
    def list_themes(cls) -> List[str]:
        """List available themes"""
        return list(cls.THEMES.keys())


class VisualizationBuilder:
    """Builder for complex visualizations"""

    def __init__(self, factory: BaseVisualizationFactory):
        """
        Initialize builder

        Args:
            factory: Visualization factory
        """
        self.factory = factory
        self.viz_type: Optional[str] = None
        self.data: Optional[pd.DataFrame] = None
        self.params: Dict[str, Any] = {}
        self.theme: Optional[VisualizationTheme] = None

    def with_type(self, viz_type: str) -> 'VisualizationBuilder':
        """Set visualization type"""
        self.viz_type = viz_type
        return self

    def with_data(self, data: pd.DataFrame) -> 'VisualizationBuilder':
        """Set data"""
        self.data = data
        return self

    def with_params(self, **kwargs) -> 'VisualizationBuilder':
        """Set parameters"""
        self.params.update(kwargs)
        return self

    def with_theme(self, theme: str) -> 'VisualizationBuilder':
        """Set theme"""
        self.theme = VisualizationTheme(theme)
        return self

    def build(self) -> Any:
        """
        Build the visualization

        Returns:
            Figure object

        Raises:
            ValueError: If required fields not set
        """
        if self.viz_type is None:
            raise ValueError("Visualization type not set")
        if self.data is None:
            raise ValueError("Data not set")

        # Create visualization
        figure = self.factory.create(self.viz_type, self.data, **self.params)

        # Apply theme if set
        if self.theme:
            figure = self.theme.apply(figure)

        return figure


class ChartRecommender:
    """Recommend appropriate chart types for data"""

    @staticmethod
    def recommend(data: pd.DataFrame, analysis_type: str = 'exploratory') -> List[Dict[str, Any]]:
        """
        Recommend chart types for data

        Args:
            data: DataFrame to analyze
            analysis_type: Type of analysis ('exploratory', 'comparison', 'distribution', 'relationship')

        Returns:
            List of recommendations with rationale
        """
        recommendations = []

        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(
            include=['object', 'category']).columns.tolist()
        datetime_cols = data.select_dtypes(
            include=['datetime64']).columns.tolist()

        # Single numeric column
        if len(numeric_cols) == 1:
            recommendations.append({
                'type': 'histogram',
                'reason': 'Single numeric column - show distribution',
                'columns': numeric_cols,
                'priority': 'high'
            })

        # Two numeric columns
        if len(numeric_cols) >= 2:
            recommendations.append({
                'type': 'scatter',
                'reason': 'Multiple numeric columns - explore relationships',
                'columns': numeric_cols[:2],
                'priority': 'high'
            })

            recommendations.append({
                'type': 'heatmap',
                'reason': 'Show correlations between numeric variables',
                'columns': numeric_cols,
                'priority': 'medium'
            })

        # Categorical and numeric
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            recommendations.append({
                'type': 'bar',
                'reason': 'Compare numeric values across categories',
                'columns': [categorical_cols[0], numeric_cols[0]],
                'priority': 'high'
            })

            recommendations.append({
                'type': 'box',
                'reason': 'Show distribution of numeric values by category',
                'columns': [categorical_cols[0], numeric_cols[0]],
                'priority': 'medium'
            })

        # Time series
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            recommendations.append({
                'type': 'line',
                'reason': 'Time series data - show trends over time',
                'columns': [datetime_cols[0], numeric_cols[0]],
                'priority': 'high'
            })

        # Single categorical
        if len(categorical_cols) > 0:
            recommendations.append({
                'type': 'pie',
                'reason': 'Show composition of categories',
                'columns': [categorical_cols[0]],
                'priority': 'medium'
            })

        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])

        return recommendations
