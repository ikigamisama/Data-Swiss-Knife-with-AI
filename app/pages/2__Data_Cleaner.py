"""
Data Cleaner Page - Automated Data Quality and Cleaning
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.helpers import detect_column_types, get_data_quality_report, detect_outliers

# Check if data is loaded
if st.session_state.get('data') is None:
    st.warning("⚠️ No data loaded. Please load data from the main page.")
    st.stop()

st.set_page_config(page_title="Data Cleaner", page_icon="🧹", layout="wide")

st.title("🧹 Data Cleaner & Quality Assessment")
st.markdown("Identify and fix data quality issues automatically")


df = st.session_state.data.copy()

# Initialize cleaned data in session state
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = df.copy()

# Sidebar - Cleaning Operations
with st.sidebar:
    st.header("🔧 Cleaning Operations")

    cleaning_mode = st.radio(
        "Mode",
        ["Interactive", "Automated", "Custom Pipeline"]
    )

    st.markdown("---")

    if cleaning_mode == "Automated":
        st.subheader("Auto-Clean Settings")
        auto_handle_missing = st.checkbox("Handle Missing Values", True)
        auto_remove_duplicates = st.checkbox("Remove Duplicates", True)
        auto_fix_dtypes = st.checkbox("Fix Data Types", True)
        auto_handle_outliers = st.checkbox("Handle Outliers", False)
        auto_standardize_text = st.checkbox("Standardize Text", True)

        if st.button("🚀 Run Auto-Clean", type="primary"):
            with st.spinner("Cleaning data..."):
                cleaned = df.copy()

                # Handle missing values
                if auto_handle_missing:
                    for col in cleaned.columns:
                        if cleaned[col].dtype in [np.float64, np.int64]:
                            cleaned[col].fillna(
                                cleaned[col].median(), inplace=True)
                        else:
                            cleaned[col].fillna(cleaned[col].mode()[0] if len(
                                cleaned[col].mode()) > 0 else 'Unknown', inplace=True)

                # Remove duplicates
                if auto_remove_duplicates:
                    cleaned = cleaned.drop_duplicates()

                # Standardize text
                if auto_standardize_text:
                    for col in cleaned.select_dtypes(include=['object']).columns:
                        cleaned[col] = cleaned[col].str.strip().str.lower()

                st.session_state.cleaned_data = cleaned
                st.success("✅ Auto-clean completed!")
                st.rerun()

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Quality Report",
    "🔍 Missing Values",
    "📌 Duplicates",
    "🎯 Outliers",
    "✅ Final Review"
])

# Tab 1: Quality Report
with tab1:
    st.subheader("Data Quality Dashboard")

    # Generate quality report
    quality_report = get_data_quality_report(df)

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Rows",
            f"{quality_report['total_rows']:,}",
            help="Number of records"
        )

    with col2:
        st.metric(
            "Total Columns",
            quality_report['total_columns'],
            help="Number of features"
        )

    with col3:
        missing_pct = quality_report['missing_values']['percentage']
        st.metric(
            "Missing Values",
            f"{missing_pct:.1f}%",
            delta=f"-{missing_pct:.1f}%" if missing_pct > 0 else "Perfect!",
            delta_color="inverse"
        )

    with col4:
        dup_count = quality_report['duplicates']
        st.metric(
            "Duplicates",
            dup_count,
            delta=f"-{dup_count}" if dup_count > 0 else "None",
            delta_color="inverse"
        )

    with col5:
        st.metric(
            "Memory Usage",
            f"{quality_report['memory_usage_mb']:.2f} MB"
        )

    st.markdown("---")

    # Detailed column analysis
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Column Statistics")

        col_stats = []
        for col, stats in quality_report['column_stats'].items():
            col_stats.append({
                'Column': col,
                'Type': stats['dtype'],
                'Non-Null': stats['non_null'],
                'Null': stats['null'],
                'Unique': stats['unique'],
                'Unique %': f"{stats['unique_ratio']*100:.1f}%"
            })

        col_df = pd.DataFrame(col_stats)
        st.dataframe(col_df, width='stretch', height=400)

    with col2:
        st.subheader("📊 Missing Values by Column")

        missing_data = quality_report['missing_values']['by_column']
        if missing_data:
            missing_df = pd.DataFrame(
                list(missing_data.items()),
                columns=['Column', 'Missing Count']
            )
            missing_df['Percentage'] = (
                missing_df['Missing Count'] / len(df) * 100).round(2)
            missing_df = missing_df.sort_values(
                'Missing Count', ascending=False)

            fig = px.bar(
                missing_df,
                x='Missing Count',
                y='Column',
                orientation='h',
                title="Missing Values Distribution",
                text='Percentage',
                color='Percentage',
                color_continuous_scale='Reds'
            )
            fig.update_traces(
                texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig, width='stretch')
        else:
            st.success("✅ No missing values found!")

    # Data type distribution
    st.subheader("🔢 Data Type Distribution")

    type_counts = df.dtypes.value_counts()

    col1, col2 = st.columns([1, 2])

    with col1:
        type_df = pd.DataFrame({
            'Data Type': type_counts.index.astype(str),
            'Count': type_counts.values
        })
        st.dataframe(type_df, width='stretch')

    with col2:
        fig = px.pie(
            type_df,
            values='Count',
            names='Data Type',
            title='Column Types Distribution',
            hole=0.4
        )
        st.plotly_chart(fig, width='stretch')

# Tab 2: Missing Values
with tab2:
    st.subheader("🔍 Missing Values Analysis & Treatment")

    missing_summary = df.isnull().sum()
    cols_with_missing = missing_summary[missing_summary > 0].index.tolist()

    if len(cols_with_missing) == 0:
        st.success("✅ No missing values found in the dataset!")
    else:
        st.warning(f"Found missing values in {len(cols_with_missing)} columns")

        # Visualization
        col1, col2 = st.columns([2, 1])

        with col1:
            # Heatmap of missing values
            fig = go.Figure(data=go.Heatmap(
                z=df[cols_with_missing].isnull().T.values,
                x=df.index[:min(500, len(df))],
                y=cols_with_missing,
                colorscale='Reds',
                showscale=False
            ))
            fig.update_layout(
                title="Missing Values Pattern (first 500 rows)",
                xaxis_title="Row Index",
                yaxis_title="Columns",
                height=400
            )
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.write("**Missing Value Counts:**")
            for col in cols_with_missing:
                count = missing_summary[col]
                pct = (count / len(df) * 100)
                st.metric(col, f"{count} ({pct:.1f}%)")

        st.markdown("---")

        # Treatment options
        st.subheader("💊 Treatment Options")

        selected_col = st.selectbox(
            "Select column to treat",
            cols_with_missing
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            strategy = st.selectbox(
                "Treatment Strategy",
                ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode",
                    "Forward Fill", "Backward Fill", "Fill with Custom Value"]
            )

        with col2:
            if strategy == "Fill with Custom Value":
                custom_value = st.text_input("Custom Value", "0")

        with col3:
            st.write("")
            st.write("")
            if st.button("Apply Treatment", type="primary"):
                cleaned = st.session_state.cleaned_data.copy()

                if strategy == "Drop Rows":
                    cleaned = cleaned.dropna(subset=[selected_col])
                elif strategy == "Fill with Mean":
                    cleaned[selected_col].fillna(
                        cleaned[selected_col].mean(), inplace=True)
                elif strategy == "Fill with Median":
                    cleaned[selected_col].fillna(
                        cleaned[selected_col].median(), inplace=True)
                elif strategy == "Fill with Mode":
                    mode_val = cleaned[selected_col].mode()[0] if len(
                        cleaned[selected_col].mode()) > 0 else 0
                    cleaned[selected_col].fillna(mode_val, inplace=True)
                elif strategy == "Forward Fill":
                    cleaned[selected_col].fillna(method='ffill', inplace=True)
                elif strategy == "Backward Fill":
                    cleaned[selected_col].fillna(method='bfill', inplace=True)
                elif strategy == "Fill with Custom Value":
                    cleaned[selected_col].fillna(custom_value, inplace=True)

                st.session_state.cleaned_data = cleaned
                st.success(f"✅ Applied {strategy} to {selected_col}")
                st.rerun()

# Tab 3: Duplicates
with tab3:
    st.subheader("📌 Duplicate Rows Detection")

    duplicates = df.duplicated()
    dup_count = duplicates.sum()

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.metric("Duplicate Rows", dup_count)

    with col2:
        st.metric("Duplicate %", f"{(dup_count/len(df)*100):.2f}%")

    with col3:
        if dup_count > 0:
            if st.button("🗑️ Remove All Duplicates", type="primary"):
                st.session_state.cleaned_data = st.session_state.cleaned_data.drop_duplicates()
                st.success(f"Removed {dup_count} duplicates!")
                st.rerun()

    if dup_count > 0:
        st.markdown("---")

        # Show duplicate rows
        st.subheader("Duplicate Records")
        dup_rows = df[duplicates]

        st.dataframe(dup_rows.head(50), width='stretch')

        # Subset duplicate detection
        st.markdown("---")
        st.subheader("Check Duplicates by Subset")

        subset_cols = st.multiselect(
            "Select columns to check for duplicates",
            df.columns.tolist(),
            default=df.columns.tolist()[:3]
        )

        if subset_cols:
            subset_dups = df.duplicated(subset=subset_cols)
            subset_dup_count = subset_dups.sum()

            st.info(
                f"Found {subset_dup_count} duplicates based on selected columns")

            if subset_dup_count > 0:
                st.dataframe(df[subset_dups].head(
                    20), width='stretch')
    else:
        st.success("✅ No duplicate rows found!")

# Tab 4: Outliers
with tab4:
    st.subheader("🎯 Outlier Detection & Treatment")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for outlier detection")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            selected_col = st.selectbox("Select Column", numeric_cols)

            method = st.radio(
                "Detection Method",
                ["IQR (Interquartile Range)", "Z-Score", "Both"]
            )

            if method in ["IQR (Interquartile Range)", "Both"]:
                iqr_threshold = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)

            if method in ["Z-Score", "Both"]:
                zscore_threshold = st.slider(
                    "Z-Score Threshold", 2.0, 4.0, 3.0, 0.1)

        with col2:
            # Detect outliers
            data_series = df[selected_col].dropna()

            outliers_mask = np.zeros(len(data_series), dtype=bool)

            if method in ["IQR (Interquartile Range)", "Both"]:
                Q1 = data_series.quantile(0.25)
                Q3 = data_series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_threshold * IQR
                upper_bound = Q3 + iqr_threshold * IQR
                iqr_outliers = (data_series < lower_bound) | (
                    data_series > upper_bound)
                outliers_mask |= iqr_outliers

            if method in ["Z-Score", "Both"]:
                from scipy import stats
                z_scores = np.abs(stats.zscore(data_series))
                zscore_outliers = z_scores > zscore_threshold
                outliers_mask |= zscore_outliers

            n_outliers = outliers_mask.sum()

            # Box plot
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=data_series,
                name=selected_col,
                boxmean='sd'
            ))

            if n_outliers > 0:
                outlier_points = data_series[outliers_mask]
                fig.add_trace(go.Scatter(
                    y=outlier_points,
                    mode='markers',
                    name='Outliers',
                    marker=dict(color='red', size=8, symbol='x')
                ))

            fig.update_layout(
                title=f"Box Plot with Outliers - {selected_col}",
                yaxis_title=selected_col,
                showlegend=True
            )
            st.plotly_chart(fig, width='stretch')

        # Outlier summary
        st.markdown("---")

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            st.metric("Outliers Found", n_outliers)
            st.metric("Outlier %", f"{(n_outliers/len(data_series)*100):.2f}%")

        with col2:
            if n_outliers > 0:
                treatment = st.selectbox(
                    "Treatment",
                    ["Remove", "Cap (Winsorize)", "Transform (Log)", "Keep"]
                )

                if st.button("Apply Treatment", type="primary"):
                    cleaned = st.session_state.cleaned_data.copy()

                    if treatment == "Remove":
                        cleaned = cleaned[~cleaned[selected_col].isin(
                            data_series[outliers_mask])]
                    elif treatment == "Cap (Winsorize)":
                        Q1 = cleaned[selected_col].quantile(0.01)
                        Q99 = cleaned[selected_col].quantile(0.99)
                        cleaned[selected_col] = cleaned[selected_col].clip(
                            Q1, Q99)
                    elif treatment == "Transform (Log)":
                        cleaned[selected_col] = np.log1p(
                            cleaned[selected_col].clip(lower=0))

                    st.session_state.cleaned_data = cleaned
                    st.success(f"✅ Applied {treatment} treatment")
                    st.rerun()

        with col3:
            if n_outliers > 0:
                st.write("**Outlier Values:**")
                outlier_df = pd.DataFrame({
                    'Index': data_series[outliers_mask].index,
                    'Value': data_series[outliers_mask].values
                }).head(20)
                st.dataframe(outlier_df, width='stretch')

# Tab 5: Final Review
with tab5:
    st.subheader("✅ Review Cleaned Data")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.write("### Before vs After Comparison")

    with col2:
        if st.button("💾 Apply Changes", type="primary"):
            st.session_state.data = st.session_state.cleaned_data.copy()
            st.success("✅ Changes applied to main dataset!")
            st.balloons()

    # Comparison metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rows",
            len(st.session_state.cleaned_data),
            delta=len(st.session_state.cleaned_data) - len(df),
            delta_color="normal"
        )

    with col2:
        original_missing = df.isnull().sum().sum()
        cleaned_missing = st.session_state.cleaned_data.isnull().sum().sum()
        st.metric(
            "Missing Values",
            cleaned_missing,
            delta=cleaned_missing - original_missing,
            delta_color="inverse"
        )

    with col3:
        original_dups = df.duplicated().sum()
        cleaned_dups = st.session_state.cleaned_data.duplicated().sum()
        st.metric(
            "Duplicates",
            cleaned_dups,
            delta=cleaned_dups - original_dups,
            delta_color="inverse"
        )

    with col4:
        original_mem = df.memory_usage(deep=True).sum() / 1024**2
        cleaned_mem = st.session_state.cleaned_data.memory_usage(
            deep=True).sum() / 1024**2
        st.metric(
            "Memory (MB)",
            f"{cleaned_mem:.2f}",
            delta=f"{cleaned_mem - original_mem:.2f}"
        )

    # Preview cleaned data
    st.markdown("---")
    st.write("### Preview Cleaned Data")
    st.dataframe(st.session_state.cleaned_data.head(
        100), width='stretch')

    # Download options
    st.markdown("---")
    st.write("### Export Cleaned Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = st.session_state.cleaned_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download CSV",
            csv,
            "cleaned_data.csv",
            "text/csv"
        )

    with col2:
        # Excel download would require openpyxl
        st.info("📊 Excel export available with openpyxl installed")

    with col3:
        # JSON download
        json_str = st.session_state.cleaned_data.to_json(
            orient='records', indent=2)
        st.download_button(
            "📥 Download JSON",
            json_str,
            "cleaned_data.json",
            "application/json"
        )
