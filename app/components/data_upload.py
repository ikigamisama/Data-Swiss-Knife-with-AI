import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional, List


class DataUploader:
    """Reusable data upload component"""

    def __init__(self,
                 accepted_formats: List[str] = None,
                 max_size_mb: int = 200):
        """
        Initialize data uploader

        Args:
            accepted_formats: List of accepted file extensions
            max_size_mb: Maximum file size in MB
        """
        self.accepted_formats = accepted_formats or [
            'csv', 'xlsx', 'json', 'parquet']
        self.max_size_mb = max_size_mb

    def render(self, key: str = "uploader") -> Optional[pd.DataFrame]:
        """
        Render upload component

        Args:
            key: Unique key for widget

        Returns:
            Loaded DataFrame or None
        """
        st.subheader("📤 Upload Data")

        # Upload method selection
        upload_method = st.radio(
            "Select Method",
            ["File Upload", "URL", "Sample Data"],
            horizontal=True,
            key=f"{key}_method"
        )

        df = None

        if upload_method == "File Upload":
            df = self._render_file_upload(key)
        elif upload_method == "URL":
            df = self._render_url_input(key)
        else:
            df = self._render_sample_data(key)

        return df

    def _render_file_upload(self, key: str) -> Optional[pd.DataFrame]:
        """Render file upload widget"""
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=self.accepted_formats,
            key=f"{key}_file",
            help=f"Max size: {self.max_size_mb}MB"
        )

        if uploaded_file is not None:
            try:
                # Show file info
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Name:** {uploaded_file.name}")
                with col2:
                    size_mb = uploaded_file.size / (1024 * 1024)
                    st.info(f"**Size:** {size_mb:.2f} MB")

                # Load based on extension
                extension = Path(uploaded_file.name).suffix.lower()

                if extension == '.csv':
                    df = pd.read_csv(uploaded_file)
                elif extension in ['.xlsx', '.xls']:
                    df = pd.read_excel(uploaded_file)
                elif extension == '.json':
                    df = pd.read_json(uploaded_file)
                elif extension == '.parquet':
                    df = pd.read_parquet(uploaded_file)
                else:
                    st.error(f"Unsupported format: {extension}")
                    return None

                st.success(
                    f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
                return df

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None

        return None

    def _render_url_input(self, key: str) -> Optional[pd.DataFrame]:
        """Render URL input widget"""
        url = st.text_input(
            "Enter URL",
            placeholder="https://example.com/data.csv",
            key=f"{key}_url"
        )

        if url and st.button("Load from URL", key=f"{key}_url_btn"):
            try:
                # Detect format from URL
                if url.endswith('.csv'):
                    df = pd.read_csv(url)
                elif url.endswith('.json'):
                    df = pd.read_json(url)
                else:
                    # Try CSV by default
                    df = pd.read_csv(url)

                st.success(f"✅ Loaded {len(df)} rows from URL")
                return df

            except Exception as e:
                st.error(f"Error loading from URL: {str(e)}")

        return None

    def _render_sample_data(self, key: str) -> Optional[pd.DataFrame]:
        """Render sample data selector"""
        samples = {
            "Iris": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
            "Tips": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
            "Titanic": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
            "Diamonds": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"
        }

        selected = st.selectbox(
            "Choose Sample Dataset",
            list(samples.keys()),
            key=f"{key}_sample"
        )

        if st.button("Load Sample", key=f"{key}_sample_btn"):
            try:
                df = pd.read_csv(samples[selected])
                st.success(f"✅ Loaded {selected} dataset")
                return df
            except Exception as e:
                st.error(f"Error loading sample: {str(e)}")

        return None
