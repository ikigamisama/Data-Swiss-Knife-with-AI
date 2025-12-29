import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd

from utils.helpers import format_number, make_pyarrow_friendly, detect_column_types
from llm.ollama_client import get_ollama_client


# Page configuration
st.set_page_config(
    page_title="Data Analysis Swiss Knife",
    page_icon="🔪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-box {
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'llm_client' not in st.session_state:
    try:
        st.session_state.llm_client = get_ollama_client()
    except:
        st.session_state.llm_client = None
if 'data_history' not in st.session_state:
    st.session_state.data_history = []
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}


def main():
    """Main application entry point"""

    # Header
    st.markdown('<h1 class="main-header">🔪 Data Swiss Knife</h1>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.image(
            "https://via.placeholder.com/150x150.png?text=Swiss+Knife", width=150)
        st.title("Navigation")


        # LLM Status
        st.subheader("🤖 AI Assistant Status")
        if st.session_state.llm_client and st.session_state.llm_client.is_available():
            st.success("✅ Ollama Connected")
            models = st.session_state.llm_client.list_models()
            selected_model = st.selectbox(
                "Select Model",
                models,
                index=models.index(
                    "gpt-oss-120b-cloud") if "gpt-oss-120b-cloud" in models else 0
            )
            st.session_state.llm_client.config.model = selected_model
        else:
            st.error("❌ Ollama Not Connected")
            if st.button("Retry Connection"):
                try:
                    st.session_state.llm_client = get_ollama_client()
                    st.rerun()
                except:
                    st.error("Failed to connect. Is Ollama running?")

        st.markdown("---")

        # Data Upload Section
        st.subheader("📂 Load Data")
        upload_method = st.radio(
            "Choose Method",
            ["Upload File", "Sample Data", "Database", "API"]
        )

        if upload_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'json', 'parquet'],
                help="Supported formats: CSV, Excel, JSON, Parquet"
            )

            if uploaded_file:
                if st.button("Load Data"):
                    with st.spinner("Loading data..."):
                        st.session_state.data = load_file(uploaded_file)
                        st.success(
                            f"✅ Loaded {len(st.session_state.data)} rows")
                        st.rerun()

        elif upload_method == "Sample Data":
            sample_datasets = {
                "Iris": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
                "Tips": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
                "Titanic": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
            }
            selected_sample = st.selectbox(
                "Select Dataset", list(sample_datasets.keys()))

            if st.button("Load Sample"):
                with st.spinner("Loading sample data..."):
                    st.session_state.data = pd.read_csv(
                        sample_datasets[selected_sample])
                    st.success(f"✅ Loaded {selected_sample} dataset")
                    st.rerun()

        elif upload_method == "Database":
            st.info("🚧 Database connection coming soon")
            db_type = st.selectbox(
                "Database Type", ["PostgreSQL", "MySQL", "MongoDB", "SQLite"])
            host = st.text_input("Host", "localhost")
            port = st.number_input("Port", value=5432)
            database = st.text_input("Database Name")

        elif upload_method == "API":
            st.info("🚧 API integration coming soon")
            api_url = st.text_input("API Endpoint")
            auth_type = st.selectbox(
                "Authentication", ["None", "API Key", "OAuth"])

    # Main Content Area
    if st.session_state.data is None:
        show_landing_page()
    else:
        show_data_overview()


def load_file(uploaded_file):
    """Load file based on extension"""
    file_extension = Path(uploaded_file.name).suffix.lower()

    if file_extension == '.csv':
        return pd.read_csv(uploaded_file)
    elif file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(uploaded_file)
    elif file_extension == '.json':
        return pd.read_json(uploaded_file)
    elif file_extension == '.parquet':
        return pd.read_parquet(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def show_landing_page():
    """Display landing page when no data is loaded"""

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>📊 Data Explorer</h3>
            <p>Explore, filter, and analyze your datasets with interactive tools</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>🤖 AI Assistant</h3>
            <p>Natural language queries powered by Ollama LLM</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>⚙️ ETL Pipelines</h3>
            <p>Build and execute data transformation pipelines</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Features Grid
    st.subheader("🎯 Key Features")

    features = {
        "Data Analysis": ["Statistical Analysis", "Correlation Analysis", "Distribution Analysis", "Outlier Detection"],
        "Data Engineering": ["ETL Pipelines", "Data Cleaning", "Feature Engineering", "Data Validation"],
        "Machine Learning": ["Auto ML", "Model Training", "Hyperparameter Tuning", "Model Evaluation"],
        "Visualization": ["Interactive Charts", "Dashboards", "Custom Plots", "Export Reports"],
        "AI Powered": ["Code Generation", "Query Translation", "Data Insights", "Anomaly Detection"],
        "Data Quality": ["Profiling", "Validation Rules", "Quality Metrics", "Monitoring"]
    }

    cols = st.columns(3)
    for idx, (category, items) in enumerate(features.items()):
        with cols[idx % 3]:
            with st.expander(f"📌 {category}"):
                for item in items:
                    st.markdown(f"✓ {item}")

    st.markdown("---")

    # Quick Start Guide
    st.subheader("🚀 Quick Start")
    st.markdown("""
    1. **Load Your Data** - Upload a file or connect to a database from the sidebar
    2. **Explore** - Navigate to Data Explorer to understand your data
    3. **Clean** - Use Data Cleaner to handle missing values and outliers
    4. **Analyze** - Run statistical analysis and generate insights
    5. **Visualize** - Create interactive charts and dashboards
    6. **AI Assist** - Ask questions in natural language using the AI Assistant
    """)


def show_data_overview():
    """Display overview of loaded data"""
    df = st.session_state.data

    # Quick Stats
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Rows", format_number(len(df)))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", format_number(missing))
    with col4:
        memory = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory", f"{memory:.2f} MB")
    with col5:
        duplicates = df.duplicated().sum()
        st.metric("Duplicates", format_number(duplicates))

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Preview", "📊 Summary", "🔍 Column Info", "🤖 AI Insights"])

    with tab1:
        st.subheader("Data Preview")
        n_rows = st.slider("Number of rows", 5, 100, 10)
        st.dataframe(make_pyarrow_friendly(df.head(n_rows)), width='stretch')

    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(make_pyarrow_friendly(df.describe()), width='stretch')

    with tab3:
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null': df.count().values,
            'Null': df.isnull().sum().values,
            'Unique': [df[col].nunique() for col in df.columns] 
        })
        st.dataframe(make_pyarrow_friendly(col_info), width='stretch')

    with tab4:
        if st.session_state.llm_client and st.session_state.llm_client.is_available():
            if st.button("🤖 Generate AI Insights"):
                with st.spinner("Analyzing data with AI..."):
                    summary = {
                        'shape': df.shape,
                        'dtypes': df.dtypes.to_dict(),
                        'missing': df.isnull().sum().to_dict(),
                        'stats': df.describe().to_dict()
                    }
                    insights = st.session_state.llm_client.explain_data(
                        summary)
                    st.markdown(insights)
        else:
            st.warning(
                "⚠️ AI Assistant not available. Connect to Ollama in the sidebar.")


if __name__ == "__main__":
    main()
