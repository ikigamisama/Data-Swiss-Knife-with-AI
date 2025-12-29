import pandas as pd
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from base_viz import BaseVisualization
import logging

logger = logging.getLogger(__name__)


class Dashboard:
    """Dashboard container for multiple visualizations"""

    def __init__(self, title: str = "Dashboard", layout: str = "grid"):
        """
        Initialize dashboard

        Args:
            title: Dashboard title
            layout: Layout type ('grid', 'tabs', 'sidebar')
        """
        self.title = title
        self.layout = layout
        self.charts: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def add_chart(self, chart: Any, title: str, position: Optional[Dict] = None):
        """
        Add chart to dashboard

        Args:
            chart: Chart figure
            title: Chart title
            position: Position dict with row, col info
        """
        self.charts.append({
            'chart': chart,
            'title': title,
            'position': position or {}
        })
        logger.info(f"Added chart: {title}")

    def remove_chart(self, index: int):
        """Remove chart by index"""
        if 0 <= index < len(self.charts):
            removed = self.charts.pop(index)
            logger.info(f"Removed chart: {removed['title']}")

    def get_chart_count(self) -> int:
        """Get number of charts"""
        return len(self.charts)

    def clear(self):
        """Clear all charts"""
        self.charts.clear()
        logger.info("Dashboard cleared")


class PlotlyDashboard(Dashboard):
    """Plotly-based dashboard"""

    def build(self, rows: int, cols: int, **kwargs) -> go.Figure:
        """
        Build Plotly dashboard

        Args:
            rows: Number of rows
            cols: Number of columns
            **kwargs: Additional subplot parameters

        Returns:
            Plotly Figure
        """
        if not self.charts:
            raise ValueError("No charts to display")

        # Create subplots
        subplot_titles = [chart['title'] for chart in self.charts]

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            **kwargs
        )

        # Add each chart
        for idx, chart_info in enumerate(self.charts):
            row = idx // cols + 1
            col = idx % cols + 1

            chart = chart_info['chart']

            # Add traces from chart to subplot
            if hasattr(chart, 'data'):
                for trace in chart.data:
                    fig.add_trace(trace, row=row, col=col)

        # Update layout
        fig.update_layout(
            title_text=self.title,
            height=kwargs.get('height', 300 * rows),
            showlegend=kwargs.get('showlegend', True)
        )

        logger.info(f"Built dashboard with {len(self.charts)} charts")
        return fig


class DashboardTemplate:
    """Pre-built dashboard templates"""

    @staticmethod
    def exploratory_analysis(data: pd.DataFrame) -> PlotlyDashboard:
        """Create exploratory analysis dashboard"""
        import plotly.express as px

        dashboard = PlotlyDashboard("Exploratory Analysis")

        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_cols) > 0:
            # Histogram
            fig1 = px.histogram(data, x=numeric_cols[0], title="Distribution")
            dashboard.add_chart(fig1, "Distribution")

            if len(numeric_cols) > 1:
                # Scatter
                fig2 = px.scatter(data, x=numeric_cols[0], y=numeric_cols[1])
                dashboard.add_chart(fig2, "Relationship")

                # Correlation heatmap
                corr = data[numeric_cols].corr()
                fig3 = px.imshow(corr, text_auto=True)
                dashboard.add_chart(fig3, "Correlations")

        return dashboard

    @staticmethod
    def time_series_analysis(data: pd.DataFrame, date_col: str, value_col: str) -> PlotlyDashboard:
        """Create time series dashboard"""
        import plotly.express as px

        dashboard = PlotlyDashboard("Time Series Analysis")

        # Line chart
        fig1 = px.line(data, x=date_col, y=value_col)
        dashboard.add_chart(fig1, "Trend")

        # Distribution
        fig2 = px.histogram(data, x=value_col)
        dashboard.add_chart(fig2, "Distribution")

        # Box plot by period
        data['period'] = pd.to_datetime(data[date_col]).dt.to_period('M')
        fig3 = px.box(data, x='period', y=value_col)
        dashboard.add_chart(fig3, "Monthly Distribution")

        return dashboard

    @staticmethod
    def comparison_dashboard(data: pd.DataFrame, category_col: str, value_col: str) -> PlotlyDashboard:
        """Create comparison dashboard"""
        import plotly.express as px

        dashboard = PlotlyDashboard("Comparison Analysis")

        # Bar chart
        fig1 = px.bar(data, x=category_col, y=value_col)
        dashboard.add_chart(fig1, "Comparison")

        # Pie chart
        agg_data = data.groupby(category_col)[value_col].sum().reset_index()
        fig2 = px.pie(agg_data, names=category_col, values=value_col)
        dashboard.add_chart(fig2, "Composition")

        # Box plot
        fig3 = px.box(data, x=category_col, y=value_col)
        dashboard.add_chart(fig3, "Distribution")

        return dashboard


class InteractiveDashboard:
    """Interactive dashboard with callbacks"""

    def __init__(self, title: str = "Interactive Dashboard"):
        self.title = title
        self.charts: List[Dict] = []
        self.callbacks: List[Dict] = []

    def add_chart(self, chart: Any, chart_id: str, **kwargs):
        """Add chart with ID for callbacks"""
        self.charts.append({
            'id': chart_id,
            'chart': chart,
            'config': kwargs
        })

    def add_callback(self, source_id: str, target_id: str, callback_fn):
        """Add interaction callback between charts"""
        self.callbacks.append({
            'source': source_id,
            'target': target_id,
            'callback': callback_fn
        })


class DashboardExporter:
    """Export dashboards to various formats"""

    @staticmethod
    def to_html(dashboard: Dashboard, filepath: str):
        """Export dashboard to HTML"""
        if isinstance(dashboard, PlotlyDashboard):
            fig = dashboard.build(rows=2, cols=2)
            fig.write_html(filepath)
            logger.info(f"Exported dashboard to {filepath}")

    @staticmethod
    def to_image(dashboard: Dashboard, filepath: str, **kwargs):
        """Export dashboard to image"""
        if isinstance(dashboard, PlotlyDashboard):
            fig = dashboard.build(rows=2, cols=2)
            fig.write_image(filepath, **kwargs)
            logger.info(f"Exported dashboard to {filepath}")

    @staticmethod
    def to_pdf(dashboard: Dashboard, filepath: str):
        """Export dashboard to PDF"""
        # Implementation depends on requirements
        logger.info(f"PDF export not yet implemented")
