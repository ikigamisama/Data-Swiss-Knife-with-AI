import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime


st.set_page_config(page_title="ETL Pipeline", page_icon="⚙️", layout="wide")

st.title("⚙️ ETL Pipeline Builder")
st.markdown("Build and execute data transformation pipelines visually")

# Check if data is loaded
if st.session_state.get('data') is None:
    st.warning("⚠️ No data loaded. Please load data from the main page.")
    st.stop()

df = st.session_state.data

# Initialize pipeline in session state
if 'pipeline_steps' not in st.session_state:
    st.session_state.pipeline_steps = []

if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = []

# Available transformations
TRANSFORMATIONS = {
    "Filter Rows": {
        "description": "Filter rows based on conditions",
        "params": ["column", "operator", "value"]
    },
    "Select Columns": {
        "description": "Select specific columns",
        "params": ["columns"]
    },
    "Drop Columns": {
        "description": "Remove specific columns",
        "params": ["columns"]
    },
    "Fill Missing": {
        "description": "Fill missing values",
        "params": ["column", "method", "value"]
    },
    "Drop Missing": {
        "description": "Drop rows with missing values",
        "params": ["columns", "how"]
    },
    "Rename Columns": {
        "description": "Rename columns",
        "params": ["old_name", "new_name"]
    },
    "Create Column": {
        "description": "Create new calculated column",
        "params": ["new_column", "expression"]
    },
    "Group By": {
        "description": "Group by columns and aggregate",
        "params": ["group_columns", "agg_column", "agg_function"]
    },
    "Sort": {
        "description": "Sort by columns",
        "params": ["columns", "ascending"]
    },
    "Change Type": {
        "description": "Change column data type",
        "params": ["column", "new_type"]
    },
    "String Operation": {
        "description": "Perform string operations",
        "params": ["column", "operation"]
    },
    "Math Operation": {
        "description": "Perform mathematical operations",
        "params": ["column", "operation", "value"]
    },
    "Date Operation": {
        "description": "Extract date components",
        "params": ["column", "component"]
    },
    "Merge Data": {
        "description": "Merge with another dataset",
        "params": ["other_data", "on", "how"]
    },
    "Pivot": {
        "description": "Pivot table transformation",
        "params": ["index", "columns", "values", "agg_func"]
    }
}

# Sidebar - Pipeline Configuration
with st.sidebar:
    st.header("🔧 Pipeline Configuration")

    pipeline_name = st.text_input("Pipeline Name", "My ETL Pipeline")

    st.markdown("---")
    st.subheader("➕ Add Transformation")

    transform_type = st.selectbox(
        "Select Transformation",
        list(TRANSFORMATIONS.keys())
    )

    st.info(TRANSFORMATIONS[transform_type]["description"])

    # Dynamic parameter inputs based on transformation type
    params = {}

    if transform_type == "Filter Rows":
        params['column'] = st.selectbox("Column", df.columns.tolist())
        params['operator'] = st.selectbox(
            "Operator", ["==", "!=", ">", "<", ">=", "<=", "contains", "startswith", "endswith"])
        params['value'] = st.text_input("Value")

    elif transform_type in ["Select Columns", "Drop Columns"]:
        params['columns'] = st.multiselect("Columns", df.columns.tolist())

    elif transform_type == "Fill Missing":
        params['column'] = st.selectbox("Column", df.columns.tolist())
        params['method'] = st.selectbox(
            "Method", ["mean", "median", "mode", "forward fill", "backward fill", "custom"])
        if params['method'] == "custom":
            params['value'] = st.text_input("Custom Value")

    elif transform_type == "Drop Missing":
        params['columns'] = st.multiselect(
            "Columns (leave empty for all)", df.columns.tolist())
        params['how'] = st.selectbox("How", ["any", "all"])

    elif transform_type == "Rename Columns":
        params['old_name'] = st.selectbox("Old Name", df.columns.tolist())
        params['new_name'] = st.text_input("New Name")

    elif transform_type == "Create Column":
        params['new_column'] = st.text_input("New Column Name")
        params['expression'] = st.text_area(
            "Expression (use column names)", "df['col1'] + df['col2']")

    elif transform_type == "Group By":
        params['group_columns'] = st.multiselect(
            "Group By Columns", df.columns.tolist())
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        params['agg_column'] = st.selectbox("Aggregate Column", numeric_cols)
        params['agg_function'] = st.selectbox(
            "Function", ["sum", "mean", "median", "count", "min", "max", "std"])

    elif transform_type == "Sort":
        params['columns'] = st.multiselect("Sort By", df.columns.tolist())
        params['ascending'] = st.checkbox("Ascending", True)

    elif transform_type == "Change Type":
        params['column'] = st.selectbox("Column", df.columns.tolist())
        params['new_type'] = st.selectbox(
            "New Type", ["int", "float", "str", "datetime", "category"])

    elif transform_type == "String Operation":
        string_cols = df.select_dtypes(include=['object']).columns.tolist()
        params['column'] = st.selectbox("Column", string_cols)
        params['operation'] = st.selectbox(
            "Operation", ["lowercase", "uppercase", "strip", "replace", "extract"])
        if params['operation'] == "replace":
            params['find'] = st.text_input("Find")
            params['replace_with'] = st.text_input("Replace With")

    elif transform_type == "Math Operation":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        params['column'] = st.selectbox("Column", numeric_cols)
        params['operation'] = st.selectbox(
            "Operation", ["add", "subtract", "multiply", "divide", "log", "sqrt", "power"])
        if params['operation'] not in ["log", "sqrt"]:
            params['value'] = st.number_input("Value", value=0.0)

    elif transform_type == "Date Operation":
        params['column'] = st.selectbox("Column", df.columns.tolist())
        params['component'] = st.selectbox(
            "Extract", ["year", "month", "day", "hour", "minute", "dayofweek", "quarter"])

    elif transform_type == "Pivot":
        params['index'] = st.multiselect("Index Columns", df.columns.tolist())
        params['columns'] = st.selectbox("Pivot Column", df.columns.tolist())
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        params['values'] = st.selectbox("Values Column", numeric_cols)
        params['agg_func'] = st.selectbox(
            "Aggregation", ["sum", "mean", "count", "min", "max"])

    if st.button("➕ Add Step", type="primary"):
        step = {
            'id': len(st.session_state.pipeline_steps) + 1,
            'type': transform_type,
            'params': params,
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.pipeline_steps.append(step)
        st.success(f"Added: {transform_type}")
        st.rerun()

    st.markdown("---")

    # Pipeline actions
    if len(st.session_state.pipeline_steps) > 0:
        if st.button("🗑️ Clear Pipeline"):
            st.session_state.pipeline_steps = []
            st.session_state.pipeline_results = []
            st.rerun()

        if st.button("💾 Save Pipeline"):
            pipeline_json = json.dumps({
                'name': pipeline_name,
                'steps': st.session_state.pipeline_steps
            }, indent=2)
            st.download_button(
                "Download Pipeline JSON",
                pipeline_json,
                f"{pipeline_name.replace(' ', '_')}.json",
                "application/json"
            )

# Main content
if len(st.session_state.pipeline_steps) == 0:
    st.info("👆 Add transformation steps from the sidebar to build your pipeline")

    # Quick start templates
    st.markdown("---")
    st.subheader("🚀 Quick Start Templates")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Data Cleaning Pipeline**")
        st.code("""
1. Drop Missing Values
2. Remove Duplicates
3. Fill Missing (numeric)
4. Standardize Text
        """)
        if st.button("Use This Template", key="template1"):
            st.info("Template feature coming soon!")

    with col2:
        st.write("**Aggregation Pipeline**")
        st.code("""
1. Group By Category
2. Aggregate Metrics
3. Sort Results
4. Export Report
        """)
        if st.button("Use This Template", key="template2"):
            st.info("Template feature coming soon!")

    with col3:
        st.write("**Feature Engineering**")
        st.code("""
1. Create Derived Columns
2. Extract Date Features
3. Normalize Values
4. One-Hot Encoding
        """)
        if st.button("Use This Template", key="template3"):
            st.info("Template feature coming soon!")

else:
    # Display pipeline
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📋 Pipeline: {pipeline_name}")
        st.write(f"**Steps:** {len(st.session_state.pipeline_steps)}")

    with col2:
        execute_pipeline = st.button(
            "▶️ Execute Pipeline", type="primary", width='stretch')
        preview_mode = st.checkbox("Preview Mode (first 1000 rows)")

    st.markdown("---")

    # Display steps
    for idx, step in enumerate(st.session_state.pipeline_steps):
        with st.expander(f"Step {idx+1}: {step['type']}", expanded=True):
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.write(f"**Type:** {step['type']}")
                st.write(f"**Parameters:**")
                for key, value in step['params'].items():
                    st.write(f"  • {key}: {value}")

            with col2:
                if st.button("✏️ Edit", key=f"edit_{idx}"):
                    st.info("Edit feature coming soon!")

            with col3:
                if st.button("🗑️ Delete", key=f"delete_{idx}"):
                    st.session_state.pipeline_steps.pop(idx)
                    st.rerun()

    # Execute pipeline
    if execute_pipeline:
        with st.spinner("⚙️ Executing pipeline..."):
            try:
                result_df = df.copy()

                if preview_mode:
                    result_df = result_df.head(1000)

                execution_log = []

                for idx, step in enumerate(st.session_state.pipeline_steps):
                    step_type = step['type']
                    params = step['params']

                    initial_rows = len(result_df)
                    initial_cols = len(result_df.columns)

                    # Execute transformation
                    if step_type == "Filter Rows":
                        col = params['column']
                        op = params['operator']
                        val = params['value']

                        if op == "==":
                            result_df = result_df[result_df[col] == val]
                        elif op == "!=":
                            result_df = result_df[result_df[col] != val]
                        elif op == ">":
                            result_df = result_df[result_df[col] > float(val)]
                        elif op == "<":
                            result_df = result_df[result_df[col] < float(val)]
                        elif op == ">=":
                            result_df = result_df[result_df[col] >= float(val)]
                        elif op == "<=":
                            result_df = result_df[result_df[col] <= float(val)]
                        elif op == "contains":
                            result_df = result_df[result_df[col].astype(
                                str).str.contains(val, na=False)]

                    elif step_type == "Select Columns":
                        result_df = result_df[params['columns']]

                    elif step_type == "Drop Columns":
                        result_df = result_df.drop(columns=params['columns'])

                    elif step_type == "Fill Missing":
                        col = params['column']
                        method = params['method']

                        if method == "mean":
                            result_df[col].fillna(
                                result_df[col].mean(), inplace=True)
                        elif method == "median":
                            result_df[col].fillna(
                                result_df[col].median(), inplace=True)
                        elif method == "mode":
                            result_df[col].fillna(
                                result_df[col].mode()[0], inplace=True)
                        elif method == "forward fill":
                            result_df[col].fillna(method='ffill', inplace=True)
                        elif method == "backward fill":
                            result_df[col].fillna(method='bfill', inplace=True)
                        elif method == "custom":
                            result_df[col].fillna(
                                params['value'], inplace=True)

                    elif step_type == "Drop Missing":
                        cols = params['columns'] if params['columns'] else None
                        result_df = result_df.dropna(
                            subset=cols, how=params['how'])

                    elif step_type == "Rename Columns":
                        result_df = result_df.rename(
                            columns={params['old_name']: params['new_name']})

                    elif step_type == "Group By":
                        result_df = result_df.groupby(params['group_columns'])[
                            params['agg_column']].agg(params['agg_function']).reset_index()

                    elif step_type == "Sort":
                        result_df = result_df.sort_values(
                            params['columns'], ascending=params['ascending'])

                    elif step_type == "String Operation":
                        col = params['column']
                        op = params['operation']

                        if op == "lowercase":
                            result_df[col] = result_df[col].str.lower()
                        elif op == "uppercase":
                            result_df[col] = result_df[col].str.upper()
                        elif op == "strip":
                            result_df[col] = result_df[col].str.strip()

                    elif step_type == "Math Operation":
                        col = params['column']
                        op = params['operation']

                        if op == "add":
                            result_df[col] = result_df[col] + params['value']
                        elif op == "subtract":
                            result_df[col] = result_df[col] - params['value']
                        elif op == "multiply":
                            result_df[col] = result_df[col] * params['value']
                        elif op == "divide":
                            result_df[col] = result_df[col] / params['value']
                        elif op == "log":
                            result_df[col] = np.log(result_df[col])
                        elif op == "sqrt":
                            result_df[col] = np.sqrt(result_df[col])

                    # Log step execution
                    execution_log.append({
                        'step': idx + 1,
                        'type': step_type,
                        'rows_before': initial_rows,
                        'rows_after': len(result_df),
                        'cols_before': initial_cols,
                        'cols_after': len(result_df.columns),
                        'status': 'Success'
                    })

                # Store results
                st.session_state.pipeline_results = {
                    'data': result_df,
                    'log': execution_log,
                    'timestamp': datetime.now().isoformat()
                }

                st.success("✅ Pipeline executed successfully!")

            except Exception as e:
                st.error(f"❌ Pipeline execution failed: {str(e)}")
                st.code(str(e))

    # Display results
    if st.session_state.pipeline_results:
        st.markdown("---")
        st.subheader("📊 Pipeline Results")

        results = st.session_state.pipeline_results
        result_df = results['data']

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Input Rows", len(df))
        with col2:
            st.metric("Output Rows", len(result_df),
                      delta=len(result_df) - len(df))
        with col3:
            st.metric("Input Columns", len(df.columns))
        with col4:
            st.metric("Output Columns", len(result_df.columns),
                      delta=len(result_df.columns) - len(df.columns))

        # Execution log
        st.markdown("---")
        st.subheader("📝 Execution Log")

        log_df = pd.DataFrame(results['log'])
        st.dataframe(log_df, width='stretch')

        # Preview results
        st.markdown("---")
        st.subheader("🔍 Output Data Preview")

        st.dataframe(result_df.head(100), width='stretch')

        # Export options
        st.markdown("---")
        st.subheader("💾 Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                csv,
                f"{pipeline_name.replace(' ', '_')}_output.csv",
                "text/csv",
                width='stretch'
            )

        with col2:
            if st.button("💾 Save to Session", width='stretch'):
                st.session_state.data = result_df
                st.success("✅ Saved to main session!")

        with col3:
            json_str = result_df.to_json(orient='records', indent=2)
            st.download_button(
                "📥 Download JSON",
                json_str,
                f"{pipeline_name.replace(' ', '_')}_output.json",
                "application/json",
                width='stretch'
            )
