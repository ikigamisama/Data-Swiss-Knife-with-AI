
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

st.title("📊 Data Explorer")
st.markdown("Explore and visualize your data interactively")

# Check if data is loaded
if st.session_state.get('data') is None:
    st.warning("⚠️ No data loaded. Please load data from the main page.")
    st.stop()

df = st.session_state.data

# Sidebar filters
with st.sidebar:
    st.header("🔧 Filters & Options")

    # Column selector
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Select Columns to Display",
        all_columns,
        default=all_columns[:10] if len(all_columns) > 10 else all_columns
    )

    # Row filter
    st.subheader("Row Filtering")
    row_start = st.number_input("Start Row", 0, len(df)-1, 0)
    row_end = st.number_input("End Row", 1, len(df), min(100, len(df)))

    # Value filters
    st.subheader("Value Filters")
    filter_column = st.selectbox("Filter Column", ["None"] + all_columns)

    if filter_column != "None":
        unique_values = df[filter_column].unique()
        if len(unique_values) <= 50:
            filter_values = st.multiselect(
                f"Select {filter_column} values",
                unique_values,
                default=unique_values.tolist()
            )
        else:
            filter_type = st.selectbox(
                "Filter Type", ["Contains", "Greater Than", "Less Than", "Equals"])
            filter_value = st.text_input("Filter Value")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Data Table", "📈 Visualizations", "📊 Distributions", "🔗 Relationships"])

# Apply filters
filtered_df = df.copy()
if filter_column != "None":
    if len(df[filter_column].unique()) <= 50 and 'filter_values' in locals():
        filtered_df = filtered_df[filtered_df[filter_column].isin(
            filter_values)]
    elif 'filter_value' in locals() and filter_value:
        if filter_type == "Contains":
            filtered_df = filtered_df[filtered_df[filter_column].astype(
                str).str.contains(filter_value, na=False)]
        elif filter_type == "Equals":
            filtered_df = filtered_df[filtered_df[filter_column]
                                      == filter_value]
        elif filter_type == "Greater Than":
            try:
                filtered_df = filtered_df[filtered_df[filter_column] > float(
                    filter_value)]
            except:
                st.error("Invalid numeric value")
        elif filter_type == "Less Than":
            try:
                filtered_df = filtered_df[filtered_df[filter_column] < float(
                    filter_value)]
            except:
                st.error("Invalid numeric value")

# Apply row slicing
display_df = filtered_df.iloc[row_start:row_end][selected_columns] if selected_columns else filtered_df.iloc[row_start:row_end]

# Tab 1: Data Table
with tab1:
    st.subheader(f"Data Table ({len(filtered_df)} rows)")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        search_term = st.text_input("🔍 Search across all columns")
    with col2:
        sort_column = st.selectbox("Sort by", ["None"] + selected_columns)
    with col3:
        sort_order = st.radio(
            "Order", ["Ascending", "Descending"], horizontal=True)

    # Apply search
    if search_term:
        mask = display_df.astype(str).apply(lambda x: x.str.contains(
            search_term, case=False, na=False)).any(axis=1)
        display_df = display_df[mask]

    # Apply sorting
    if sort_column != "None":
        display_df = display_df.sort_values(
            sort_column, ascending=(sort_order == "Ascending"))

    # Display with formatting
    st.dataframe(
        display_df,
        width='stretch',
        height=500
    )

    # Download button
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Filtered Data",
        csv,
        "filtered_data.csv",
        "text/csv",
        key='download-csv'
    )

# Tab 2: Visualizations
with tab2:
    st.subheader("Interactive Visualizations")

    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Scatter Plot", "Line Chart", "Bar Chart",
            "Box Plot", "Heatmap", "3D Scatter"]
    )

    numeric_cols = filtered_df.select_dtypes(
        include=[np.number]).columns.tolist()
    categorical_cols = filtered_df.select_dtypes(
        include=['object', 'category']).columns.tolist()

    if viz_type == "Scatter Plot":
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X Axis", numeric_cols)
            y_axis = st.selectbox(
                "Y Axis", [c for c in numeric_cols if c != x_axis])
        with col2:
            color_by = st.selectbox("Color By (optional)", [
                                    "None"] + categorical_cols + numeric_cols)
            size_by = st.selectbox("Size By (optional)", [
                                   "None"] + numeric_cols)

        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color=color_by if color_by != "None" else None,
            size=size_by if size_by != "None" else None,
            hover_data=selected_columns[:5],
            title=f"{y_axis} vs {x_axis}"
        )
        st.plotly_chart(fig, width='stretch')

    elif viz_type == "Line Chart":
        x_axis = st.selectbox("X Axis", all_columns)
        y_axes = st.multiselect("Y Axes", numeric_cols,
                                default=numeric_cols[:1])

        fig = go.Figure()
        for y in y_axes:
            fig.add_trace(go.Scatter(
                x=filtered_df[x_axis], y=filtered_df[y], mode='lines', name=y))

        fig.update_layout(title="Line Chart",
                          xaxis_title=x_axis, yaxis_title="Value")
        st.plotly_chart(fig, width='stretch')

    elif viz_type == "Bar Chart":
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox(
                "X Axis (Category)", categorical_cols if categorical_cols else all_columns)
            y_axis = st.selectbox(
                "Y Axis (Value)", numeric_cols if numeric_cols else all_columns)
        with col2:
            agg_func = st.selectbox(
                "Aggregation", ["sum", "mean", "median", "count", "min", "max"])
            color_by = st.selectbox("Color By", ["None"] + categorical_cols)

        agg_df = filtered_df.groupby(
            x_axis)[y_axis].agg(agg_func).reset_index()
        fig = px.bar(
            agg_df,
            x=x_axis,
            y=y_axis,
            title=f"{agg_func.capitalize()} of {y_axis} by {x_axis}",
            color=color_by if color_by != "None" else None
        )
        st.plotly_chart(fig, width='stretch')

    elif viz_type == "Box Plot":
        y_axes = st.multiselect("Select Numeric Columns",
                                numeric_cols, default=numeric_cols[:3])
        group_by = st.selectbox("Group By (optional)", [
                                "None"] + categorical_cols)

        fig = go.Figure()
        if group_by != "None":
            for col in y_axes:
                for group in filtered_df[group_by].unique():
                    data = filtered_df[filtered_df[group_by] == group][col]
                    fig.add_trace(go.Box(y=data, name=f"{col} - {group}"))
        else:
            for col in y_axes:
                fig.add_trace(go.Box(y=filtered_df[col], name=col))

        fig.update_layout(title="Box Plot")
        st.plotly_chart(fig, width='stretch')

    elif viz_type == "Heatmap":
        selected_numeric = st.multiselect(
            "Select Columns for Correlation",
            numeric_cols,
            default=numeric_cols[:5] if len(
                numeric_cols) >= 5 else numeric_cols
        )

        if len(selected_numeric) >= 2:
            corr_matrix = filtered_df[selected_numeric].corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap",
                color_continuous_scale="RdBu_r"
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning(
                "Select at least 2 numeric columns for correlation analysis")

    elif viz_type == "3D Scatter":
        if len(numeric_cols) >= 3:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X Axis", numeric_cols)
                y_axis = st.selectbox(
                    "Y Axis", [c for c in numeric_cols if c != x_axis])
            with col2:
                z_axis = st.selectbox(
                    "Z Axis", [c for c in numeric_cols if c not in [x_axis, y_axis]])
                color_by = st.selectbox("Color", ["None"] + categorical_cols)

            fig = px.scatter_3d(
                filtered_df,
                x=x_axis,
                y=y_axis,
                z=z_axis,
                color=color_by if color_by != "None" else None,
                title="3D Scatter Plot"
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Need at least 3 numeric columns for 3D visualization")

# Tab 3: Distributions
with tab3:
    st.subheader("Distribution Analysis")

    dist_type = st.radio("Select Analysis", [
                         "Histograms", "Density Plots", "Q-Q Plots"], horizontal=True)

    if dist_type == "Histograms":
        selected_cols = st.multiselect(
            "Select Columns",
            numeric_cols,
            default=numeric_cols[:4] if len(
                numeric_cols) >= 4 else numeric_cols
        )

        bins = st.slider("Number of Bins", 10, 100, 30)

        n_cols = 2
        n_rows = (len(selected_cols) + 1) // 2

        fig = make_subplots(rows=n_rows, cols=n_cols,
                            subplot_titles=selected_cols)

        for idx, col in enumerate(selected_cols):
            row = idx // n_cols + 1
            col_pos = idx % n_cols + 1

            fig.add_trace(
                go.Histogram(x=filtered_df[col], name=col, nbinsx=bins),
                row=row,
                col=col_pos
            )

        fig.update_layout(height=300*n_rows, showlegend=False,
                          title_text="Distribution Histograms")
        st.plotly_chart(fig, width='stretch')

    elif dist_type == "Density Plots":
        selected_col = st.selectbox("Select Column", numeric_cols)
        group_by = st.selectbox("Group By (optional)", [
                                "None"] + categorical_cols)

        fig = go.Figure()
        if group_by != "None":
            for group in filtered_df[group_by].unique():
                data = filtered_df[filtered_df[group_by]
                                   == group][selected_col].dropna()
                fig.add_trace(go.Violin(y=data, name=str(group),
                              box_visible=True, meanline_visible=True))
        else:
            data = filtered_df[selected_col].dropna()
            fig.add_trace(
                go.Violin(y=data, box_visible=True, meanline_visible=True))

        fig.update_layout(title=f"Density Plot - {selected_col}")
        st.plotly_chart(fig, width='stretch')

# Tab 4: Relationships
with tab4:
    st.subheader("Relationship Analysis")

    if len(numeric_cols) >= 2:
        st.write("### Correlation Matrix")
        corr_cols = st.multiselect(
            "Select columns for correlation",
            numeric_cols,
            default=numeric_cols
        )

        if len(corr_cols) >= 2:
            corr_matrix = filtered_df[corr_cols].corr()

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(corr_matrix.style.background_gradient(
                    cmap='coolwarm', axis=None))

            with col2:
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Correlation Heatmap"
                )
                st.plotly_chart(fig, width='stretch')

            # Find strong correlations
            st.write("### Strong Correlations (|r| > 0.7)")
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        strong_corr.append({
                            'Column 1': corr_matrix.columns[i],
                            'Column 2': corr_matrix.columns[j],
                            'Correlation': round(corr_matrix.iloc[i, j], 3)
                        })

            if strong_corr:
                st.dataframe(pd.DataFrame(strong_corr))
            else:
                st.info("No strong correlations found")
    else:
        st.warning("Need at least 2 numeric columns for correlation analysis")
