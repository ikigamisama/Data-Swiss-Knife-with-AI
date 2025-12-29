
from llm.ollama_client import get_ollama_client
import streamlit as st
import pandas as pd
import numpy as np
import sys
import traceback
import io


st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI Data Assistant")
st.markdown("Ask questions about your data in natural language")

# Check if data is loaded
if st.session_state.get('data') is None:
    st.warning("⚠️ No data loaded. Please load data from the main page.")
    st.stop()

# Check if LLM is available
if st.session_state.get('llm_client') is None or not st.session_state.llm_client.is_available():
    st.error("❌ AI Assistant not available. Please ensure Ollama is running.")
    st.info("Start Ollama with: `ollama serve` and pull the model with: `ollama pull gpt-oss-120b-cloud`")
    st.stop()

df = st.session_state.data
llm_client = st.session_state.llm_client

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'code_execution_results' not in st.session_state:
    st.session_state.code_execution_results = []

# Sidebar with data context
with st.sidebar:
    st.header("📊 Data Context")
    st.metric("Rows", len(df))
    st.metric("Columns", len(df.columns))

    st.subheader("Column Types")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=['object', 'category']).columns.tolist()

    with st.expander("Numeric Columns"):
        for col in numeric_cols:
            st.write(f"• {col}")

    with st.expander("Categorical Columns"):
        for col in categorical_cols:
            st.write(f"• {col}")

    st.markdown("---")

    st.subheader("⚙️ Settings")
    temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.1)
    auto_execute = st.checkbox("Auto-execute generated code", value=True)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.code_execution_results = []
        st.rerun()

# Main content
tab1, tab2, tab3 = st.tabs(
    ["💬 Chat", "📝 Suggested Queries", "🧪 Code Playground"])

# Tab 1: Chat Interface
with tab1:
    # Display chat history
    chat_container = st.container()

    with chat_container:
        for idx, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])

                    # Show generated code if available
                    if 'code' in message and message['code']:
                        with st.expander("📄 Generated Code"):
                            st.code(message['code'], language='python')

                        # Show execution result if available
                        if 'result' in message and message['result']:
                            with st.expander("✅ Execution Result"):
                                result = message['result']
                                if isinstance(result, pd.DataFrame):
                                    st.dataframe(result)
                                elif isinstance(result, dict) and 'figure' in result:
                                    st.plotly_chart(result['figure'])
                                else:
                                    st.write(result)

    # Chat input
    user_query = st.chat_input("Ask a question about your data...")

    if user_query:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_query
        })

        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                # Prepare data context
                df_info = {
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'numeric_columns': numeric_cols,
                    'categorical_columns': categorical_cols,
                    'sample_data': df.head(3).to_dict('records')
                }

                # Generate pandas code
                code = llm_client.nl_to_pandas(user_query, df_info)

                response_message = {
                    'role': 'assistant',
                    'content': f"I'll help you with that. Here's the analysis:",
                    'code': code
                }

                # Execute code if auto-execute is enabled
                if auto_execute and code:
                    try:
                        # Create execution environment
                        local_vars = {'df': df, 'pd': pd, 'np': np}

                        # Capture output
                        output_buffer = io.StringIO()
                        sys.stdout = output_buffer

                        # Execute code
                        exec(code, {'__builtins__': __builtins__}, local_vars)

                        # Restore stdout
                        sys.stdout = sys.__stdout__

                        # Get result
                        if 'result' in local_vars:
                            result = local_vars['result']
                        else:
                            result = output_buffer.getvalue()

                        response_message['result'] = result
                        response_message['content'] += "\n\n✅ Code executed successfully!"

                    except Exception as e:
                        response_message['content'] += f"\n\n⚠️ Execution error: {str(e)}"
                        response_message['error'] = str(e)

                st.session_state.chat_history.append(response_message)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': f"I encountered an error: {str(e)}\n\nPlease try rephrasing your question."
                })

        st.rerun()

# Tab 2: Suggested Queries
with tab2:
    st.subheader("💡 Suggested Queries")
    st.markdown("Click on any suggestion to ask the AI Assistant")

    suggestions = [
        "Show me the first 10 rows of the data",
        "What are the summary statistics for all numeric columns?",
        f"Show me the distribution of {numeric_cols[0] if numeric_cols else 'values'}",
        "Find and show any missing values in the dataset",
        "Calculate the correlation between numeric columns",
        f"Show me the top 10 values in {categorical_cols[0] if categorical_cols else 'the first column'}",
        "Identify outliers in numeric columns using z-score method",
        "Create a pivot table summarizing the data",
        "Show me rows with duplicate values",
        "What is the data type of each column?"
    ]

    # Add custom suggestions based on data
    if len(numeric_cols) >= 2:
        suggestions.append(f"Plot {numeric_cols[0]} vs {numeric_cols[1]}")

    if categorical_cols:
        suggestions.append(f"Show value counts for {categorical_cols[0]}")

    # Display suggestions in columns
    col1, col2 = st.columns(2)

    for idx, suggestion in enumerate(suggestions):
        with col1 if idx % 2 == 0 else col2:
            if st.button(suggestion, key=f"suggest_{idx}", use_container_width=True):
                # Set the query and switch to chat tab
                st.session_state.suggested_query = suggestion
                st.rerun()

    # If a suggestion was clicked, process it in the chat tab
    if st.session_state.get('suggested_query'):
        # Switch to chat tab by setting the active tab
        st.session_state.active_tab = "💬 Chat"
        # Clear the suggested query flag
        suggested_query = st.session_state.pop('suggested_query')
        
        # Add to chat history and process
        st.session_state.chat_history.append({
            'role': 'user',
            'content': suggested_query
        })
        
        # Generate response for the suggested query
        with st.spinner("🤔 Thinking..."):
            try:
                # Prepare data context
                df_info = {
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'numeric_columns': numeric_cols,
                    'categorical_columns': categorical_cols,
                    'sample_data': df.head(3).to_dict('records')
                }

                # Generate pandas code
                code = llm_client.nl_to_pandas(suggested_query, df_info)

                response_message = {
                    'role': 'assistant',
                    'content': f"I'll help you with that. Here's the analysis:",
                    'code': code
                }

                # Execute code if auto-execute is enabled
                if auto_execute and code:
                    try:
                        # Create execution environment
                        local_vars = {'df': df, 'pd': pd, 'np': np}

                        # Capture output
                        output_buffer = io.StringIO()
                        sys.stdout = output_buffer

                        # Execute code
                        exec(code, {'__builtins__': __builtins__}, local_vars)

                        # Restore stdout
                        sys.stdout = sys.__stdout__

                        # Get result
                        if 'result' in local_vars:
                            result = local_vars['result']
                        else:
                            result = output_buffer.getvalue()

                        response_message['result'] = result
                        response_message['content'] += "\n\n✅ Code executed successfully!"

                    except Exception as e:
                        response_message['content'] += f"\n\n⚠️ Execution error: {str(e)}"
                        response_message['error'] = str(e)

                st.session_state.chat_history.append(response_message)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': f"I encountered an error: {str(e)}\n\nPlease try rephrasing your question."
                })

# Tab 3: Code Playground
with tab3:
    st.subheader("🧪 Code Playground")
    st.markdown("Write and execute custom Python code on your data")

    # Code editor
    default_code = """# Available variables:
# - df: Your dataframe
# - pd: pandas module
# - np: numpy module

# Example: Get summary statistics
result = df.describe()

# Set 'result' variable to display output
"""

    code_input = st.text_area(
        "Python Code",
        value=default_code,
        height=300,
        help="Write pandas code to analyze your data. Set 'result' variable to display output."
    )

    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        if st.button("▶️ Run Code", type="primary"):
            try:
                # Create execution environment
                local_vars = {
                    'df': df.copy(),
                    'pd': pd,
                    'np': np
                }

                # Capture output
                output_buffer = io.StringIO()
                sys.stdout = output_buffer

                # Execute code
                exec(code_input, {'__builtins__': __builtins__}, local_vars)

                # Restore stdout
                sys.stdout = sys.__stdout__

                # Display results
                st.success("✅ Code executed successfully!")

                if 'result' in local_vars:
                    result = local_vars['result']
                    st.subheader("Result:")

                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result, width='stretch')
                    elif isinstance(result, (pd.Series, np.ndarray)):
                        st.write(result)
                    else:
                        st.write(result)

                # Show printed output
                output = output_buffer.getvalue()
                if output:
                    st.subheader("Output:")
                    st.text(output)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.code(traceback.format_exc())

    with col2:
        if st.button("🤖 Optimize Code"):
            with st.spinner("Optimizing..."):
                try:
                    optimized = llm_client.optimize_query(code_input, "pandas")
                    st.subheader("AI Suggestions:")
                    st.markdown(optimized)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Code templates
    st.markdown("---")
    st.subheader("📚 Code Templates")

    templates = {
        "Filter Data": """# Filter rows based on condition
result = df[df['column_name'] > value]""",

        "Group and Aggregate": """# Group by column and aggregate
result = df.groupby('category_column')['numeric_column'].agg(['mean', 'sum', 'count'])""",

        "Handle Missing Values": """# Fill missing values
df_clean = df.fillna(df.mean())
result = df_clean""",

        "Create New Column": """# Create calculated column
df['new_column'] = df['col1'] + df['col2']
result = df""",

        "Merge DataFrames": """# Merge two dataframes
# df2 = pd.read_csv('other_file.csv')
# result = pd.merge(df, df2, on='key_column', how='left')""",

        "Pivot Table": """# Create pivot table
result = df.pivot_table(
    values='value_column',
    index='row_column',
    columns='col_column',
    aggfunc='mean'
)""",
    }

    template_cols = st.columns(3)
    for idx, (name, code) in enumerate(templates.items()):
        with template_cols[idx % 3]:
            with st.expander(name):
                st.code(code, language='python')
                if st.button(f"Use Template", key=f"template_{idx}"):
                    st.session_state.template_code = code
                    st.rerun()

    if 'template_code' in st.session_state:
        st.info(f"Template copied! Paste it in the code editor above.")
        del st.session_state.template_code
