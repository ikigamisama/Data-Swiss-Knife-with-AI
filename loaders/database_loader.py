import pandas as pd
from typing import Any, Optional, Dict, List
from ..core.base import DataLoader
import logging

logger = logging.getLogger(__name__)


class SQLDatabaseLoader(DataLoader):
    """SQL Database loader using SQLAlchemy"""

    def __init__(self, connection_string: str):
        """
        Initialize SQL database loader

        Args:
            connection_string: SQLAlchemy connection string
                Examples:
                - PostgreSQL: postgresql://user:password@localhost:5432/dbname
                - MySQL: mysql://user:password@localhost:3306/dbname
                - SQLite: sqlite:///path/to/database.db
        """
        self.connection_string = connection_string
        self.engine = None

    def validate(self, source: Any) -> bool:
        """Validate database connection"""
        try:
            from sqlalchemy import create_engine
            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database validation failed: {e}")
            return False

    def load(self, source: Any, **kwargs) -> pd.DataFrame:
        """
        Load data from SQL database

        Args:
            source: SQL query string or table name
            **kwargs: Additional parameters (chunksize, index_col, etc.)

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            from sqlalchemy import create_engine

            if self.engine is None:
                self.engine = create_engine(self.connection_string)

            # Determine if source is a query or table name
            if source.strip().upper().startswith('SELECT'):
                logger.info(f"Executing SQL query")
                df = pd.read_sql_query(source, self.engine, **kwargs)
            else:
                logger.info(f"Loading table: {source}")
                df = pd.read_sql_table(source, self.engine, **kwargs)

            logger.info(
                f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Error loading from database: {e}")
            raise RuntimeError(f"Failed to load from database: {str(e)}")

    def get_tables(self) -> List[str]:
        """Get list of tables in database"""
        try:
            from sqlalchemy import create_engine, inspect

            if self.engine is None:
                self.engine = create_engine(self.connection_string)

            inspector = inspect(self.engine)
            return inspector.get_table_names()

        except Exception as e:
            logger.error(f"Error getting tables: {e}")
            return []

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a table"""
        try:
            from sqlalchemy import create_engine, inspect

            if self.engine is None:
                self.engine = create_engine(self.connection_string)

            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name)

            return {
                'table_name': table_name,
                'columns': [
                    {
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col['nullable']
                    }
                    for col in columns
                ]
            }

        except Exception as e:
            logger.error(f"Error getting table schema: {e}")
            return {}

    def execute_query(self, query: str) -> Any:
        """Execute a SQL query without returning data"""
        try:
            from sqlalchemy import create_engine

            if self.engine is None:
                self.engine = create_engine(self.connection_string)

            with self.engine.connect() as conn:
                result = conn.execute(query)
                conn.commit()

            return result

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise RuntimeError(f"Failed to execute query: {str(e)}")

    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


class MongoDBLoader(DataLoader):
    """MongoDB loader"""

    def __init__(self, connection_string: str, database: str):
        """
        Initialize MongoDB loader

        Args:
            connection_string: MongoDB connection string
            database: Database name
        """
        self.connection_string = connection_string
        self.database_name = database
        self.client = None
        self.db = None

    def validate(self, source: Any) -> bool:
        """Validate MongoDB connection"""
        try:
            from pymongo import MongoClient

            client = MongoClient(self.connection_string,
                                 serverSelectionTimeoutMS=5000)
            client.server_info()
            return True

        except Exception as e:
            logger.error(f"MongoDB validation failed: {e}")
            return False

    def load(self, source: Any, query: Optional[Dict] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from MongoDB collection

        Args:
            source: Collection name
            query: MongoDB query filter (default: {})
            **kwargs: Additional parameters (projection, limit, skip, etc.)

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            from pymongo import MongoClient

            if self.client is None:
                self.client = MongoClient(self.connection_string)
                self.db = self.client[self.database_name]

            collection = self.db[source]

            # Set defaults
            query = query or {}
            projection = kwargs.get('projection', None)
            limit = kwargs.get('limit', 0)
            skip = kwargs.get('skip', 0)

            # Query collection
            cursor = collection.find(query, projection).skip(skip).limit(limit)

            # Convert to DataFrame
            data = list(cursor)
            df = pd.DataFrame(data)

            # Remove MongoDB _id if present and not needed
            if '_id' in df.columns and not kwargs.get('include_id', False):
                df = df.drop('_id', axis=1)

            logger.info(
                f"Successfully loaded {len(df)} documents from {source}")
            return df

        except Exception as e:
            logger.error(f"Error loading from MongoDB: {e}")
            raise RuntimeError(f"Failed to load from MongoDB: {str(e)}")

    def get_collections(self) -> List[str]:
        """Get list of collections in database"""
        try:
            from pymongo import MongoClient

            if self.client is None:
                self.client = MongoClient(self.connection_string)
                self.db = self.client[self.database_name]

            return self.db.list_collection_names()

        except Exception as e:
            logger.error(f"Error getting collections: {e}")
            return []

    def aggregate(self, collection: str, pipeline: List[Dict]) -> pd.DataFrame:
        """Run aggregation pipeline"""
        try:
            if self.client is None:
                from pymongo import MongoClient
                self.client = MongoClient(self.connection_string)
                self.db = self.client[self.database_name]

            coll = self.db[collection]
            result = coll.aggregate(pipeline)

            df = pd.DataFrame(list(result))
            return df

        except Exception as e:
            logger.error(f"Error running aggregation: {e}")
            raise RuntimeError(f"Failed to run aggregation: {str(e)}")

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


class RedisLoader(DataLoader):
    """Redis key-value store loader"""

    def __init__(self, host: str = 'localhost', port: int = 6379,
                 password: Optional[str] = None, db: int = 0):
        """
        Initialize Redis loader

        Args:
            host: Redis host
            port: Redis port
            password: Redis password (if required)
            db: Database number
        """
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.client = None

    def validate(self, source: Any) -> bool:
        """Validate Redis connection"""
        try:
            import redis

            client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                socket_timeout=5
            )
            client.ping()
            return True

        except Exception as e:
            logger.error(f"Redis validation failed: {e}")
            return False

    def load(self, source: Any, pattern: str = '*', **kwargs) -> pd.DataFrame:
        """
        Load data from Redis

        Args:
            source: Key pattern to match
            pattern: Pattern for scanning keys (default: '*')
            **kwargs: Additional parameters

        Returns:
            pd.DataFrame: Loaded data with keys and values
        """
        try:
            import redis

            if self.client is None:
                self.client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    db=self.db
                )

            # Scan for keys matching pattern
            keys = []
            for key in self.client.scan_iter(match=pattern):
                keys.append(key.decode('utf-8'))

            # Get values for all keys
            data = []
            for key in keys:
                value = self.client.get(key)
                if value:
                    data.append({
                        'key': key,
                        'value': value.decode('utf-8')
                    })

            df = pd.DataFrame(data)
            logger.info(f"Successfully loaded {len(df)} keys from Redis")
            return df

        except Exception as e:
            logger.error(f"Error loading from Redis: {e}")
            raise RuntimeError(f"Failed to load from Redis: {str(e)}")

    def close(self):
        """Close Redis connection"""
        if self.client:
            self.client.close()
            logger.info("Redis connection closed")


class DuckDBLoader(DataLoader):
    """DuckDB embedded database loader"""

    def __init__(self, database: str = ':memory:'):
        """
        Initialize DuckDB loader

        Args:
            database: Database path or ':memory:' for in-memory
        """
        self.database = database
        self.conn = None

    def validate(self, source: Any) -> bool:
        """Validate DuckDB connection"""
        try:
            import duckdb
            conn = duckdb.connect(self.database)
            conn.close()
            return True
        except Exception as e:
            logger.error(f"DuckDB validation failed: {e}")
            return False

    def load(self, source: Any, **kwargs) -> pd.DataFrame:
        """
        Load data using DuckDB SQL query

        Args:
            source: SQL query string
            **kwargs: Additional parameters

        Returns:
            pd.DataFrame: Query results
        """
        try:
            import duckdb

            if self.conn is None:
                self.conn = duckdb.connect(self.database)

            # Execute query and convert to DataFrame
            result = self.conn.execute(source)
            df = result.df()

            logger.info(f"Successfully executed query, got {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Error loading from DuckDB: {e}")
            raise RuntimeError(f"Failed to load from DuckDB: {str(e)}")

    def register_dataframe(self, name: str, df: pd.DataFrame):
        """Register a DataFrame as a DuckDB table"""
        try:
            import duckdb

            if self.conn is None:
                self.conn = duckdb.connect(self.database)

            self.conn.register(name, df)
            logger.info(f"Registered DataFrame as table: {name}")

        except Exception as e:
            logger.error(f"Error registering DataFrame: {e}")
            raise RuntimeError(f"Failed to register DataFrame: {str(e)}")

    def close(self):
        """Close DuckDB connection"""
        if self.conn:
            self.conn.close()
            logger.info("DuckDB connection closed")
