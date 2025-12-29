import streamlit as st
import plotly.express as px
from typing import Optional, Dict, Any


class VizSelector:
    """Visualization type selector and configurator"""

    VIZ_TYPES = {
        "Scatter Plot": "scatter",
        "Line Chart": "line",
        "Bar Chart": "bar",
        "Histogram": "histogram",
        "Box Plot": "box",
        "Heatmap": "heatmap",
        "Pie Chart": "pie",
        "3D Scatter": "scatter_3d"
    }

    def render(self, df, key: str = "viz_selector") -> Optional[Any]:
        """
        Render visualization selector

        Args:
            df: DataFrame to visualize
            key: Unique key for widget

        Returns:
            Plotly figure or None
        """
        st.subheader("📊 Visualization")

        # Select visualization type
        viz_type = st.selectbox(
            "Chart Type",
            list(self.VIZ_TYPES.keys()),
            key=f"{key}_type"
        )

        # Get configuration for selected type
        config = self._get_config(viz_type, df, key)

        # Create visualization button
        if st.button("📈 Create Chart", type="primary", key=f"{key}_create"):
            return self._create_visualization(viz_type, df, config)

        return None

    def _get_config(self, viz_type: str, df, key: str) -> Dict[str, Any]:
        """Get configuration for visualization type"""
        config = {}

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=['object', 'category']).columns.tolist()
        all_cols = df.columns.tolist()

        col1, col2 = st.columns(2)

        if viz_type in ["Scatter Plot", "Line Chart"]:
            with col1:
                config['x'] = st.selectbox("X Axis", all_cols, key=f"{key}_x")
            with col2:
                config['y'] = st.selectbox(
                    "Y Axis", numeric_cols, key=f"{key}_y")
            config['color'] = st.selectbox(
                "Color By", ["None"] + all_cols, key=f"{key}_color")

        elif viz_type == "Bar Chart":
            with col1:
                config['x'] = st.selectbox("X Axis", all_cols, key=f"{key}_x")
            with col2:
                config['y'] = st.selectbox(
                    "Y Axis", numeric_cols, key=f"{key}_y")

        elif viz_type == "Histogram":
            config['x'] = st.selectbox("Column", numeric_cols, key=f"{key}_x")
            config['nbins'] = st.slider("Bins", 10, 100, 30, key=f"{key}_bins")

        elif viz_type == "Box Plot":
            config['y'] = st.multiselect(
                "Columns", numeric_cols, default=numeric_cols[:3], key=f"{key}_y")

        elif viz_type == "Heatmap":
            config['columns'] = st.multiselect(
                "Columns for Correlation",
                numeric_cols,
                default=numeric_cols,
                key=f"{key}_cols"
            )

        elif viz_type == "Pie Chart":
            with col1:
                config['names'] = st.selectbox(
                    "Labels", categorical_cols, key=f"{key}_names")
            with col2:
                config['values'] = st.selectbox(
                    "Values", numeric_cols, key=f"{key}_values")

        elif viz_type == "3D Scatter":
            col1, col2, col3 = st.columns(3)
            with col1:
                config['x'] = st.selectbox(
                    "X Axis", numeric_cols, key=f"{key}_x")
            with col2:
                config['y'] = st.selectbox(
                    "Y Axis", numeric_cols, key=f"{key}_y")
            with col3:
                config['z'] = st.selectbox(
                    "Z Axis", numeric_cols, key=f"{key}_z")

        return config

    def _create_visualization(self, viz_type: str, df, config: Dict) -> Any:
        """Create visualization based on type and config"""
        try:
            viz_func = self.VIZ_TYPES[viz_type]

            if viz_type == "Scatter Plot":
                fig = px.scatter(df, x=config['x'], y=config['y'],
                                 color=config['color'] if config['color'] != "None" else None)
            elif viz_type == "Line Chart":
                fig = px.line(df, x=config['x'], y=config['y'],
                              color=config['color'] if config['color'] != "None" else None)
            elif viz_type == "Bar Chart":
                fig = px.bar(df, x=config['x'], y=config['y'])
            elif viz_type == "Histogram":
                fig = px.histogram(df, x=config['x'], nbins=config['nbins'])
            elif viz_type == "Box Plot":
                fig = px.box(df, y=config['y'])
            elif viz_type == "Heatmap":
                corr = df[config['columns']].corr()
                fig = px.imshow(corr, text_auto=True, aspect='auto')
            elif viz_type == "Pie Chart":
                fig = px.pie(
                    df, names=config['names'], values=config['values'])
            elif viz_type == "3D Scatter":
                fig = px.scatter_3d(
                    df, x=config['x'], y=config['y'], z=config['z'])
            else:
                return None

            return fig

        except Exception as e:
            st.error(f"Error creating visualization: {e}")
            return None
