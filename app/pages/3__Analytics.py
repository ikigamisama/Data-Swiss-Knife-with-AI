"""
Advanced Analytics Page - Statistical Analysis, Time Series, Correlations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")

st.title("📈 Advanced Analytics")
st.markdown("Statistical analysis, correlations, and insights")

# Check if data is loaded
if st.session_state.get('data') is None:
    st.warning("⚠️ No data loaded. Please load data from the main page.")
    st.stop()

df = st.session_state.data

# Sidebar
with st.sidebar:
    st.header("📊 Analysis Type")

    analysis_type = st.selectbox(
        "Select Analysis",
        [
            "Statistical Summary",
            "Correlation Analysis",
            "Distribution Analysis",
            "Time Series Analysis",
            "Hypothesis Testing",
            "PCA & Dimensionality",
            "Trend Analysis"
        ]
    )

    st.markdown("---")
    st.subheader("⚙️ Settings")

    confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
    show_annotations = st.checkbox("Show Annotations", True)

# Main content
if analysis_type == "Statistical Summary":
    st.subheader("📊 Comprehensive Statistical Summary")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for statistical analysis")
    else:
        # Basic statistics
        st.write("### Descriptive Statistics")

        stats_df = df[numeric_cols].describe().T
        stats_df['variance'] = df[numeric_cols].var()
        stats_df['skewness'] = df[numeric_cols].skew()
        stats_df['kurtosis'] = df[numeric_cols].kurtosis()

        st.dataframe(stats_df.style.background_gradient(
            cmap='YlOrRd', axis=0), width='stretch')

        # Visual summary
        st.markdown("---")
        st.write("### Visual Summary")

        selected_cols = st.multiselect(
            "Select columns for visualization",
            numeric_cols,
            default=numeric_cols[:4] if len(
                numeric_cols) >= 4 else numeric_cols
        )

        if selected_cols:
            # Create subplots
            n_cols = 2
            n_rows = (len(selected_cols) + 1) // 2

            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=[
                    f"{col} Distribution" for col in selected_cols],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            for idx, col in enumerate(selected_cols):
                row = idx // n_cols + 1
                col_pos = idx % n_cols + 1

                # Histogram
                fig.add_trace(
                    go.Histogram(
                        x=df[col],
                        name=col,
                        nbinsx=30,
                        showlegend=False
                    ),
                    row=row,
                    col=col_pos
                )

                # Add mean line
                mean_val = df[col].mean()
                fig.add_vline(
                    x=mean_val,
                    line_dash="dash",
                    line_color="red",
                    row=row,
                    col=col_pos
                )

            fig.update_layout(height=300*n_rows, showlegend=False)
            st.plotly_chart(fig, width='stretch')

        # Normality tests
        st.markdown("---")
        st.write("### Normality Tests (Shapiro-Wilk)")

        normality_results = []
        for col in numeric_cols:
            sample = df[col].dropna().sample(min(5000, len(df[col].dropna())))
            statistic, p_value = stats.shapiro(sample)

            normality_results.append({
                'Column': col,
                'Statistic': round(statistic, 4),
                'P-Value': round(p_value, 4),
                'Normal?': 'Yes' if p_value > 0.05 else 'No'
            })

        norm_df = pd.DataFrame(normality_results)
        st.dataframe(norm_df, width='stretch')

elif analysis_type == "Correlation Analysis":
    st.subheader("🔗 Correlation Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            corr_method = st.selectbox(
                "Correlation Method",
                ["Pearson", "Spearman", "Kendall"]
            )

            selected_cols = st.multiselect(
                "Select Columns",
                numeric_cols,
                default=numeric_cols
            )

            threshold = st.slider(
                "Highlight threshold",
                0.0, 1.0, 0.5, 0.05
            )

        with col2:
            if len(selected_cols) >= 2:
                # Calculate correlation
                corr_matrix = df[selected_cols].corr(
                    method=corr_method.lower())

                # Heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect='auto',
                    color_continuous_scale='RdBu_r',
                    title=f'{corr_method} Correlation Matrix',
                    zmin=-1,
                    zmax=1
                )
                st.plotly_chart(fig, width='stretch')

        # Strong correlations
        if len(selected_cols) >= 2:
            st.markdown("---")
            st.write(f"### Strong Correlations (|r| > {threshold})")

            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > threshold:
                        strong_corr.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': round(corr_val, 3),
                            'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate',
                            'Direction': 'Positive' if corr_val > 0 else 'Negative'
                        })

            if strong_corr:
                corr_df = pd.DataFrame(strong_corr).sort_values(
                    'Correlation', key=abs, ascending=False)
                st.dataframe(corr_df, width='stretch')

                # Scatter plot for top correlation
                st.markdown("---")
                st.write("### Scatter Plot - Top Correlation")

                top_corr = corr_df.iloc[0]
                fig = px.scatter(
                    df,
                    x=top_corr['Variable 1'],
                    y=top_corr['Variable 2'],
                    trendline="ols",
                    title=f"{top_corr['Variable 2']} vs {top_corr['Variable 1']} (r = {top_corr['Correlation']})"
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.info(f"No correlations above threshold {threshold}")

elif analysis_type == "Distribution Analysis":
    st.subheader("📊 Distribution Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        st.warning("No numeric columns found")
    else:
        selected_col = st.selectbox("Select Column", numeric_cols)

        col1, col2 = st.columns(2)

        with col1:
            # Histogram with KDE
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=df[selected_col].dropna(),
                name='Histogram',
                nbinsx=50,
                histnorm='probability density'
            ))

            # Add KDE
            from scipy.stats import gaussian_kde
            data = df[selected_col].dropna()
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)

            fig.add_trace(go.Scatter(
                x=x_range,
                y=kde(x_range),
                name='KDE',
                line=dict(color='red', width=2)
            ))

            fig.update_layout(
                title=f'Distribution of {selected_col}',
                xaxis_title=selected_col,
                yaxis_title='Density'
            )
            st.plotly_chart(fig, width='stretch')

        with col2:
            # Q-Q Plot
            fig = go.Figure()

            theoretical_quantiles = stats.probplot(
                df[selected_col].dropna(), dist="norm")

            fig.add_trace(go.Scatter(
                x=theoretical_quantiles[0][0],
                y=theoretical_quantiles[0][1],
                mode='markers',
                name='Data'
            ))

            fig.add_trace(go.Scatter(
                x=theoretical_quantiles[0][0],
                y=theoretical_quantiles[1][1] +
                theoretical_quantiles[1][0] * theoretical_quantiles[0][0],
                mode='lines',
                name='Normal',
                line=dict(color='red')
            ))

            fig.update_layout(
                title=f'Q-Q Plot - {selected_col}',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles'
            )
            st.plotly_chart(fig, width='stretch')

        # Statistics
        st.markdown("---")
        st.write("### Distribution Statistics")

        col1, col2, col3, col4 = st.columns(4)

        data = df[selected_col].dropna()

        with col1:
            st.metric("Mean", f"{data.mean():.2f}")
            st.metric("Std Dev", f"{data.std():.2f}")

        with col2:
            st.metric("Median", f"{data.median():.2f}")
            st.metric(
                "IQR", f"{data.quantile(0.75) - data.quantile(0.25):.2f}")

        with col3:
            st.metric("Skewness", f"{data.skew():.2f}")
            st.metric("Kurtosis", f"{data.kurtosis():.2f}")

        with col4:
            st.metric("Min", f"{data.min():.2f}")
            st.metric("Max", f"{data.max():.2f}")

elif analysis_type == "Time Series Analysis":
    st.subheader("📅 Time Series Analysis")

    # Detect datetime columns
    datetime_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        else:
            try:
                pd.to_datetime(df[col].head(100), errors='raise')
                datetime_cols.append(col)
            except:
                pass

    if len(datetime_cols) == 0:
        st.warning("No datetime columns found. Trying to detect date columns...")

        # Try to find date-like columns
        possible_date_cols = [
            col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]

        if possible_date_cols:
            st.info(f"Found possible date columns: {possible_date_cols}")
            selected_date_col = st.selectbox(
                "Select date column", possible_date_cols)

            try:
                df[selected_date_col] = pd.to_datetime(df[selected_date_col])
                datetime_cols = [selected_date_col]
            except:
                st.error("Could not convert to datetime")
                st.stop()
        else:
            st.error("No date columns found")
            st.stop()

    date_col = st.selectbox("Select Date Column", datetime_cols)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    value_col = st.selectbox("Select Value Column", numeric_cols)

    # Prepare time series
    ts_df = df[[date_col, value_col]].copy()
    ts_df[date_col] = pd.to_datetime(ts_df[date_col])
    ts_df = ts_df.sort_values(date_col)
    ts_df = ts_df.set_index(date_col)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Time series plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=ts_df.index,
            y=ts_df[value_col],
            mode='lines',
            name=value_col
        ))

        # Add moving average
        window = st.sidebar.slider("Moving Average Window", 2, 30, 7)
        ts_df['MA'] = ts_df[value_col].rolling(window=window).mean()

        fig.add_trace(go.Scatter(
            x=ts_df.index,
            y=ts_df['MA'],
            mode='lines',
            name=f'{window}-period MA',
            line=dict(dash='dash')
        ))

        fig.update_layout(
            title=f'Time Series - {value_col}',
            xaxis_title='Date',
            yaxis_title=value_col
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.write("### Time Series Stats")

        # Trend
        from scipy.stats import linregress
        x = np.arange(len(ts_df))
        slope, intercept, r_value, p_value, std_err = linregress(
            x, ts_df[value_col].dropna())

        trend = "Increasing" if slope > 0 else "Decreasing"
        st.metric("Trend", trend, f"{slope:.4f}/period")

        # Seasonality check (simple)
        st.metric("R² (Trend)", f"{r_value**2:.3f}")

        # Stationarity (simple check)
        st.metric("Mean", f"{ts_df[value_col].mean():.2f}")
        st.metric("Std Dev", f"{ts_df[value_col].std():.2f}")

    # Decomposition visualization
    st.markdown("---")
    st.write("### Seasonal Decomposition")

    try:
        from statsmodels.tsa.seasonal import seasonal_decompose

        # Resample to regular frequency if needed
        freq = st.selectbox("Frequency", ["D", "W", "M", "Q", "Y"])
        ts_resampled = ts_df[value_col].resample(freq).mean().dropna()

        if len(ts_resampled) >= 4:
            decomposition = seasonal_decompose(
                ts_resampled, model='additive', period=min(12, len(ts_resampled)//2))

            fig = make_subplots(
                rows=4,
                cols=1,
                subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
                vertical_spacing=0.08
            )

            fig.add_trace(go.Scatter(x=ts_resampled.index,
                          y=ts_resampled.values, name='Original'), row=1, col=1)
            fig.add_trace(go.Scatter(x=decomposition.trend.index,
                          y=decomposition.trend.values, name='Trend'), row=2, col=1)
            fig.add_trace(go.Scatter(x=decomposition.seasonal.index,
                          y=decomposition.seasonal.values, name='Seasonal'), row=3, col=1)
            fig.add_trace(go.Scatter(x=decomposition.resid.index,
                          y=decomposition.resid.values, name='Residual'), row=4, col=1)

            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, width='stretch')
    except Exception as e:
        st.info(f"Seasonal decomposition not available: {e}")

elif analysis_type == "Hypothesis Testing":
    st.subheader("🧪 Hypothesis Testing")

    test_type = st.selectbox(
        "Select Test",
        ["T-Test (Two Sample)", "ANOVA", "Chi-Square", "Mann-Whitney U"]
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=['object', 'category']).columns.tolist()

    if test_type == "T-Test (Two Sample)":
        st.write("Compare means of two groups")

        col1, col2 = st.columns(2)

        with col1:
            group_col = st.selectbox("Grouping Variable", categorical_cols)
            value_col = st.selectbox("Value Variable", numeric_cols)

        with col2:
            groups = df[group_col].unique()[:2]
            if len(groups) >= 2:
                st.write(f"**Groups:** {groups[0]} vs {groups[1]}")

                group1 = df[df[group_col] == groups[0]][value_col].dropna()
                group2 = df[df[group_col] == groups[1]][value_col].dropna()

                t_stat, p_value = stats.ttest_ind(group1, group2)

                st.metric("T-Statistic", f"{t_stat:.4f}")
                st.metric("P-Value", f"{p_value:.4f}")

                if p_value < 0.05:
                    st.error("✅ Significant difference (p < 0.05)")
                else:
                    st.success("❌ No significant difference (p >= 0.05)")

        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Box(y=group1, name=str(groups[0])))
        fig.add_trace(go.Box(y=group2, name=str(groups[1])))
        fig.update_layout(title=f'{value_col} by {group_col}')
        st.plotly_chart(fig, width='stretch')

elif analysis_type == "PCA & Dimensionality":
    st.subheader("🔬 Principal Component Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for PCA")
    else:
        # Prepare data
        X = df[numeric_cols].dropna()

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca = PCA()
        pca_result = pca.fit_transform(X_scaled)

        col1, col2 = st.columns([2, 1])

        with col1:
            # Scree plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                y=pca.explained_variance_ratio_ * 100,
                name='Explained Variance'
            ))

            cumsum = np.cumsum(pca.explained_variance_ratio_ * 100)
            fig.add_trace(go.Scatter(
                x=[f'PC{i+1}' for i in range(len(cumsum))],
                y=cumsum,
                name='Cumulative',
                yaxis='y2'
            ))

            fig.update_layout(
                title='Scree Plot',
                yaxis=dict(title='Explained Variance (%)'),
                yaxis2=dict(title='Cumulative (%)',
                            overlaying='y', side='right')
            )
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.write("### Variance Explained")
            for i in range(min(5, len(pca.explained_variance_ratio_))):
                st.metric(
                    f"PC{i+1}",
                    f"{pca.explained_variance_ratio_[i]*100:.1f}%"
                )

        # Biplot
        st.markdown("---")
        st.write("### PCA Biplot (PC1 vs PC2)")

        pca_df = pd.DataFrame(
            pca_result[:, :2],
            columns=['PC1', 'PC2']
        )

        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            title='PCA Biplot'
        )
        st.plotly_chart(fig, width='stretch')

elif analysis_type == "Trend Analysis":
    st.subheader("📊 Trend Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_col = st.selectbox("Select Column", numeric_cols)

    # Prepare data with index
    y = df[selected_col].dropna().values
    x = np.arange(len(y))

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    trend_line = slope * x + intercept

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name='Data'
        ))

        fig.add_trace(go.Scatter(
            x=x,
            y=trend_line,
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title=f'Trend Analysis - {selected_col}',
            xaxis_title='Index',
            yaxis_title=selected_col
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.write("### Trend Statistics")
        st.metric("Slope", f"{slope:.4f}")
        st.metric("R²", f"{r_value**2:.4f}")
        st.metric("P-Value", f"{p_value:.4f}")

        if p_value < 0.05:
            trend_direction = "Increasing" if slope > 0 else "Decreasing"
            st.success(f"✅ Significant {trend_direction} Trend")
        else:
            st.info("No significant trend")
