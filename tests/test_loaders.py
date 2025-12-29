"""
Unit Tests for Data Loaders
"""
from loaders.csv_loader import CSVLoader, TSVLoader, ExcelLoader, JSONLoader, ParquetLoader
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import io

# Import loaders
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCSVLoader:
    """Tests for CSV Loader"""

    @pytest.fixture
    def sample_csv(self):
        """Create sample CSV file"""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000]
        })

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            return f.name

    def test_validate_valid_csv(self, sample_csv):
        """Test validation of valid CSV file"""
        loader = CSVLoader()
        assert loader.validate(sample_csv) is True

    def test_validate_invalid_file(self):
        """Test validation of invalid file"""
        loader = CSVLoader()
        assert loader.validate('nonexistent.csv') is False

    def test_load_csv(self, sample_csv):
        """Test loading CSV file"""
        loader = CSVLoader()
        df = loader.load(sample_csv)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ['id', 'name', 'age', 'salary']

    def test_load_with_custom_delimiter(self):
        """Test loading CSV with custom delimiter"""
        data = "id;name;age\n1;Alice;25\n2;Bob;30"

        loader = CSVLoader(delimiter=';')
        df = loader.load(io.StringIO(data))

        assert len(df) == 2
        assert 'name' in df.columns

    def test_detect_delimiter(self):
        """Test automatic delimiter detection"""
        csv_data = "id,name,age\n1,Alice,25"
        tsv_data = "id\tname\tage\n1\tAlice\t25"

        loader = CSVLoader()

        assert loader.detect_delimiter(io.StringIO(csv_data)) == ','
        assert loader.detect_delimiter(io.StringIO(tsv_data)) == '\t'

    def test_load_chunked(self, sample_csv):
        """Test loading CSV in chunks"""
        loader = CSVLoader()
        chunks = list(loader.load_chunked(sample_csv, chunksize=2))

        assert len(chunks) == 3  # 5 rows / 2 = 3 chunks
        assert len(chunks[0]) == 2
        assert len(chunks[-1]) == 1

    def test_get_metadata(self, sample_csv):
        """Test getting metadata without loading full file"""
        loader = CSVLoader()
        metadata = loader.get_metadata(sample_csv)

        assert 'columns' in metadata
        assert 'column_count' in metadata
        assert metadata['column_count'] == 4


class TestTSVLoader:
    """Tests for TSV Loader"""

    def test_load_tsv(self):
        """Test loading TSV file"""
        data = "id\tname\tage\n1\tAlice\t25\n2\tBob\t30"

        loader = TSVLoader()
        df = loader.load(io.StringIO(data))

        assert len(df) == 2
        assert list(df.columns) == ['id', 'name', 'age']


class TestExcelLoader:
    """Tests for Excel Loader"""

    @pytest.fixture
    def sample_excel(self):
        """Create sample Excel file"""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'value': [100, 200, 300]
        })

        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            data.to_excel(f.name, index=False, sheet_name='Sheet1')
            return f.name

    def test_load_excel(self, sample_excel):
        """Test loading Excel file"""
        loader = ExcelLoader()
        df = loader.load(sample_excel)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_get_sheet_names(self, sample_excel):
        """Test getting sheet names"""
        loader = ExcelLoader()
        sheets = loader.get_sheet_names(sample_excel)

        assert 'Sheet1' in sheets


class TestJSONLoader:
    """Tests for JSON Loader"""

    def test_load_json_records(self):
        """Test loading JSON in records format"""
        data = [
            {'id': 1, 'name': 'Alice', 'age': 25},
            {'id': 2, 'name': 'Bob', 'age': 30}
        ]

        import json
        json_str = json.dumps(data)

        loader = JSONLoader()
        df = loader.load(io.StringIO(json_str), orient='records')

        assert len(df) == 2
        assert 'name' in df.columns


class TestParquetLoader:
    """Tests for Parquet Loader"""

    @pytest.fixture
    def sample_parquet(self):
        """Create sample Parquet file"""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'value': [10.5, 20.3, 30.1, 40.7, 50.2]
        })

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            data.to_parquet(f.name, index=False)
            return f.name

    def test_load_parquet(self, sample_parquet):
        """Test loading Parquet file"""
        loader = ParquetLoader()
        df = loader.load(sample_parquet)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5


# Integration tests
class TestLoaderIntegration:
    """Integration tests for loaders"""

    def test_load_different_formats(self):
        """Test loading different file formats"""
        # Create sample data
        data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Save in different formats
            csv_path = tmpdir / 'test.csv'
            json_path = tmpdir / 'test.json'
            parquet_path = tmpdir / 'test.parquet'

            data.to_csv(csv_path, index=False)
            data.to_json(json_path, orient='records')
            data.to_parquet(parquet_path, index=False)

            # Load with appropriate loaders
            csv_df = CSVLoader().load(csv_path)
            json_df = JSONLoader().load(json_path)
            parquet_df = ParquetLoader().load(parquet_path)

            # Verify all loaded correctly
            assert len(csv_df) == 3
            assert len(json_df) == 3
            assert len(parquet_df) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
