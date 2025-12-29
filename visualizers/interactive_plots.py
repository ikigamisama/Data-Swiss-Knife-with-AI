import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any, Dict, Optional, List
from ..core.base import Visualization, VisualizationFactory
import logging

logger = logging.getLogger(__name__)


class ScatterPlot(Visualization):
    """Scatter plot visualization"""

    def get_type(self) -> str:
        return "scatter"

    def create(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create scatter plot"""
        try:
            fig = px.scatter(
                data,
                x=kwargs.get('x'),
                y=kwargs.get('y'),
                color=kwargs.get('color'),
                size=kwargs.get('size'),
                hover_data=kwargs.get('hover_data'),
                title=kwargs.get('title', 'Scatter Plot'),
                trendline=kwargs.get('trendline')
            )

            fig.update_layout(
                template=kwargs.get('template', 'plotly_white'),
                height=kwargs.get('height', 500)
            )

            return fig
        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")
            raise


class LinePlot(Visualization):
    """Line plot visualization"""

    def get_type(self) -> str:
        return "line"

    def create(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create line plot"""
        try:
            fig = px.line(
                data,
                x=kwargs.get('x'),
                y=kwargs.get('y'),
                color=kwargs.get('color'),
                title=kwargs.get('title', 'Line Plot')
            )

            fig.update_layout(
                template=kwargs.get('template', 'plotly_white'),
                height=kwargs.get('height', 500)
            )

            return fig
        except Exception as e:
            logger.error(f"Error creating line plot: {e}")
            raise


class BarChart(Visualization):
    """Bar chart visualization"""

    def get_type(self) -> str:
        return "bar"

    def create(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create bar chart"""
        try:
            fig = px.bar(
                data,
                x=kwargs.get('x'),
                y=kwargs.get('y'),
                color=kwargs.get('color'),
                barmode=kwargs.get('barmode', 'group'),
                title=kwargs.get('title', 'Bar Chart')
            )

            fig.update_layout(
                template=kwargs.get('template', 'plotly_white'),
                height=kwargs.get('height', 500)
            )

            return fig
        except Exception as e:
            logger.error(f"Error creating bar chart: {e}")
            raise


class Histogram(Visualization):
    """Histogram visualization"""

    def get_type(self) -> str:
        return "histogram"

    def create(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create histogram"""
        try:
            fig = px.histogram(
                data,
                x=kwargs.get('x'),
                nbins=kwargs.get('nbins', 30),
                color=kwargs.get('color'),
                title=kwargs.get('title', 'Histogram')
            )

            fig.update_layout(
                template=kwargs.get('template', 'plotly_white'),
                height=kwargs.get('height', 500)
            )

            return fig
        except Exception as e:
            logger.error(f"Error creating histogram: {e}")
            raise


class BoxPlot(Visualization):
    """Box plot visualization"""

    def get_type(self) -> str:
        return "box"

    def create(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create box plot"""
        try:
            fig = px.box(
                data,
                x=kwargs.get('x'),
                y=kwargs.get('y'),
                color=kwargs.get('color'),
                title=kwargs.get('title', 'Box Plot')
            )

            fig.update_layout(
                template=kwargs.get('template', 'plotly_white'),
                height=kwargs.get('height', 500)
            )

            return fig
        except Exception as e:
            logger.error(f"Error creating box plot: {e}")
            raise


class HeatmapPlot(Visualization):
    """Heatmap visualization"""

    def get_type(self) -> str:
        return "heatmap"

    def create(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create heatmap"""
        try:
            # If data is not a matrix, create correlation matrix
            if data.shape[1] > 2:
                plot_data = data.corr()
            else:
                plot_data = data

            fig = px.imshow(
                plot_data,
                text_auto=kwargs.get('text_auto', True),
                aspect=kwargs.get('aspect', 'auto'),
                color_continuous_scale=kwargs.get('color_scale', 'RdBu_r'),
                title=kwargs.get('title', 'Heatmap')
            )

            fig.update_layout(
                height=kwargs.get('height', 500)
            )

            return fig
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            raise


class PieChart(Visualization):
    """Pie chart visualization"""

    def get_type(self) -> str:
        return "pie"

    def create(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create pie chart"""
        try:
            fig = px.pie(
                data,
                values=kwargs.get('values'),
                names=kwargs.get('names'),
                title=kwargs.get('title', 'Pie Chart'),
                hole=kwargs.get('hole', 0)
            )

            fig.update_layout(
                height=kwargs.get('height', 500)
            )

            return fig
        except Exception as e:
            logger.error(f"Error creating pie chart: {e}")
            raise


class ViolinPlot(Visualization):
    """Violin plot visualization"""

    def get_type(self) -> str:
        return "violin"

    def create(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create violin plot"""
        try:
            fig = px.violin(
                data,
                x=kwargs.get('x'),
                y=kwargs.get('y'),
                color=kwargs.get('color'),
                box=kwargs.get('box', True),
                title=kwargs.get('title', 'Violin Plot')
            )

            fig.update_layout(
                template=kwargs.get('template', 'plotly_white'),
                height=kwargs.get('height', 500)
            )

            return fig
        except Exception as e:
            logger.error(f"Error creating violin plot: {e}")
            raise


class PlotlyVisualizationFactory(VisualizationFactory):
    """Factory for creating Plotly visualizations"""

    def __init__(self):
        self._visualizations = {
            'scatter': ScatterPlot(),
            'line': LinePlot(),
            'bar': BarChart(),
            'histogram': Histogram(),
            'box': BoxPlot(),
            'heatmap': HeatmapPlot(),
            'pie': PieChart(),
            'violin': ViolinPlot()
        }

    def create_visualization(self, viz_type: str) -> Visualization:
        """Create visualization by type"""
        if viz_type not in self._visualizations:
            raise ValueError(f"Unknown visualization type: {viz_type}")

        return self._visualizations[viz_type]

    def list_available(self) -> List[str]:
        """List available visualization types"""
        return list(self._visualizations.keys())


class DashboardBuilder:
    """Build multi-plot dashboards"""

    def __init__(self):
        self.plots = []

    def add_plot(self, viz: Visualization, data: pd.DataFrame, **kwargs):
        """Add a plot to the dashboard"""
        self.plots.append({
            'viz': viz,
            'data': data,
            'kwargs': kwargs
        })
        return self

    def build(self, rows: int, cols: int, **layout_kwargs) -> go.Figure:
        """Build the dashboard"""
        try:
            # Create subplots
            subplot_titles = [p['kwargs'].get('title', '') for p in self.plots]
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=subplot_titles,
                **layout_kwargs
            )

            # Add each plot
            for idx, plot in enumerate(self.plots):
                row = idx // cols + 1
                col = idx % cols + 1

                # Create individual figure
                individual_fig = plot['viz'].create(
                    plot['data'], **plot['kwargs'])

                # Add traces to subplot
                for trace in individual_fig.data:
                    fig.add_trace(trace, row=row, col=col)

            # Update layout
            fig.update_layout(
                height=layout_kwargs.get('height', 300 * rows),
                showlegend=layout_kwargs.get('showlegend', True),
                template='plotly_white'
            )

            logger.info(f"Dashboard created with {len(self.plots)} plots")
            return fig

        except Exception as e:
            logger.error(f"Error building dashboard: {e}")
            raise
