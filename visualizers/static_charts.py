import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, Optional, List
from .base_viz import BaseVisualization, BaseVisualizationFactory
import logging

logger = logging.getLogger(__name__)


class MatplotlibBase(BaseVisualization):
    """Base class for Matplotlib visualizations"""

    def __init__(self):
        super().__init__()
        self.fig = None
        self.ax = None

    def _setup_figure(self, figsize: tuple = (10, 6)):
        """Setup matplotlib figure"""
        self.fig, self.ax = plt.subplots(figsize=figsize)
        return self.fig, self.ax

    def save(self, filepath: str, **kwargs):
        """Save figure to file"""
        if self.fig is None:
            raise ValueError("No figure to save")

        dpi = kwargs.get('dpi', 300)
        bbox_inches = kwargs.get('bbox_inches', 'tight')

        self.fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        logger.info(f"Saved visualization to {filepath}")


class MatplotlibHistogram(MatplotlibBase):
    """Matplotlib histogram"""

    def get_type(self) -> str:
        return "histogram"

    def create(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create histogram"""
        column = kwargs.get('column') or kwargs.get('x')
        bins = kwargs.get('bins', 30)
        title = kwargs.get('title', f'Histogram of {column}')

        self.validate_data(data, [column])

        fig, ax = self._setup_figure(kwargs.get('figsize', (10, 6)))

        ax.hist(data[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        self.figure = fig
        return fig


class MatplotlibScatter(MatplotlibBase):
    """Matplotlib scatter plot"""

    def get_type(self) -> str:
        return "scatter"

    def create(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create scatter plot"""
        x = kwargs.get('x')
        y = kwargs.get('y')
        color = kwargs.get('color')
        title = kwargs.get('title', f'{y} vs {x}')

        self.validate_data(data, [x, y])

        fig, ax = self._setup_figure(kwargs.get('figsize', (10, 6)))

        if color and color in data.columns:
            scatter = ax.scatter(data[x], data[y], c=data[color],
                                 cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=ax, label=color)
        else:
            ax.scatter(data[x], data[y], alpha=0.6)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        self.figure = fig
        return fig


class MatplotlibLine(MatplotlibBase):
    """Matplotlib line plot"""

    def get_type(self) -> str:
        return "line"

    def create(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create line plot"""
        x = kwargs.get('x')
        y = kwargs.get('y')
        title = kwargs.get('title', f'{y} over {x}')

        self.validate_data(data, [x, y])

        fig, ax = self._setup_figure(kwargs.get('figsize', (12, 6)))

        ax.plot(data[x], data[y], marker='o', linewidth=2)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels if they're dates
        if pd.api.types.is_datetime64_any_dtype(data[x]):
            plt.xticks(rotation=45, ha='right')

        self.figure = fig
        return fig


class MatplotlibBar(MatplotlibBase):
    """Matplotlib bar chart"""

    def get_type(self) -> str:
        return "bar"

    def create(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create bar chart"""
        x = kwargs.get('x')
        y = kwargs.get('y')
        title = kwargs.get('title', f'{y} by {x}')
        horizontal = kwargs.get('horizontal', False)

        self.validate_data(data, [x, y])

        fig, ax = self._setup_figure(kwargs.get('figsize', (10, 6)))

        if horizontal:
            ax.barh(data[x], data[y])
            ax.set_xlabel(y)
            ax.set_ylabel(x)
        else:
            ax.bar(data[x], data[y])
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            plt.xticks(rotation=45, ha='right')

        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y' if not horizontal else 'x')

        self.figure = fig
        return fig


class MatplotlibBox(MatplotlibBase):
    """Matplotlib box plot"""

    def get_type(self) -> str:
        return "box"

    def create(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create box plot"""
        columns = kwargs.get('columns')
        title = kwargs.get('title', 'Box Plot')

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        fig, ax = self._setup_figure(kwargs.get('figsize', (10, 6)))

        ax.boxplot([data[col].dropna() for col in columns],
                   labels=columns, patch_artist=True)
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')

        self.figure = fig
        return fig


class MatplotlibHeatmap(MatplotlibBase):
    """Matplotlib heatmap"""

    def get_type(self) -> str:
        return "heatmap"

    def create(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create heatmap"""
        title = kwargs.get('title', 'Heatmap')
        cmap = kwargs.get('cmap', 'RdYlGn')
        annot = kwargs.get('annot', True)

        # If data is not a matrix, create correlation matrix
        if data.shape[1] > 2:
            plot_data = data.select_dtypes(include=[np.number]).corr()
        else:
            plot_data = data

        fig, ax = self._setup_figure(kwargs.get('figsize', (10, 8)))

        sns.heatmap(plot_data, annot=annot, fmt='.2f', cmap=cmap,
                    center=0, square=True, ax=ax)
        ax.set_title(title)

        self.figure = fig
        return fig


class MatplotlibPie(MatplotlibBase):
    """Matplotlib pie chart"""

    def get_type(self) -> str:
        return "pie"

    def create(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create pie chart"""
        values = kwargs.get('values')
        labels = kwargs.get('labels')
        title = kwargs.get('title', 'Pie Chart')

        self.validate_data(data, [values, labels])

        fig, ax = self._setup_figure(kwargs.get('figsize', (10, 8)))

        ax.pie(data[values], labels=data[labels], autopct='%1.1f%%',
               startangle=90)
        ax.set_title(title)

        self.figure = fig
        return fig


class MatplotlibVisualizationFactory(BaseVisualizationFactory):
    """Factory for Matplotlib visualizations"""

    def _register_visualizations(self):
        """Register Matplotlib visualizations"""
        self.register('histogram', MatplotlibHistogram())
        self.register('scatter', MatplotlibScatter())
        self.register('line', MatplotlibLine())
        self.register('bar', MatplotlibBar())
        self.register('box', MatplotlibBox())
        self.register('heatmap', MatplotlibHeatmap())
        self.register('pie', MatplotlibPie())


# Convenience function
def create_static_chart(chart_type: str, data: pd.DataFrame, **kwargs) -> plt.Figure:
    """
    Create a static chart using Matplotlib

    Args:
        chart_type: Type of chart
        data: Data to visualize
        **kwargs: Chart parameters

    Returns:
        Matplotlib figure
    """
    factory = MatplotlibVisualizationFactory()
    return factory.create(chart_type, data, **kwargs)
