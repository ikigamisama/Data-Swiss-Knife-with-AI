import pandas as pd
import requests
from typing import Any, Dict, Optional, List
from core.base import DataLoader
from core.exceptions import DataLoadError
import logging

logger = logging.getLogger(__name__)


class APILoader(DataLoader):
    """Load data from REST APIs"""

    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None,
                 auth: Optional[tuple] = None, timeout: int = 30):
        """
        Initialize API loader

        Args:
            base_url: Base URL for the API
            headers: HTTP headers to include in requests
            auth: Authentication tuple (username, password)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.headers = headers or {}
        self.auth = auth
        self.timeout = timeout
        self.session = requests.Session()

    def validate(self, source: Any) -> bool:
        """Validate API endpoint is accessible"""
        try:
            url = f"{self.base_url}/{source}" if not source.startswith(
                'http') else source
            response = self.session.get(
                url,
                headers=self.headers,
                auth=self.auth,
                timeout=5
            )
            return response.status_code in [200, 201, 204]
        except Exception as e:
            logger.error(f"API validation failed: {e}")
            return False

    def load(self, source: Any, method: str = 'GET', **kwargs) -> pd.DataFrame:
        """
        Load data from API endpoint

        Args:
            source: API endpoint path or full URL
            method: HTTP method (GET, POST, etc.)
            **kwargs: Additional request parameters

        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            url = f"{self.base_url}/{source}" if not source.startswith(
                'http') else source

            logger.info(f"Loading data from API: {url}")

            # Make request
            response = self.session.request(
                method=method,
                url=url,
                headers=self.headers,
                auth=self.auth,
                timeout=self.timeout,
                **kwargs
            )

            response.raise_for_status()

            # Parse response
            data = response.json()

            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                elif 'results' in data:
                    df = pd.DataFrame(data['results'])
                else:
                    df = pd.DataFrame([data])
            else:
                raise DataLoadError(
                    f"Unexpected response format: {type(data)}")

            logger.info(f"Successfully loaded {len(df)} rows from API")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise DataLoadError(f"Failed to load from API: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading from API: {e}")
            raise DataLoadError(f"Failed to load from API: {str(e)}")

    def load_paginated(self, source: str, page_param: str = 'page',
                       max_pages: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from paginated API endpoint

        Args:
            source: API endpoint
            page_param: Parameter name for page number
            max_pages: Maximum number of pages to load

        Returns:
            pd.DataFrame: Combined data from all pages
        """
        all_data = []
        page = 1

        while True:
            if max_pages and page > max_pages:
                break

            try:
                params = {page_param: page}
                df = self.load(source, params=params)

                if df.empty:
                    break

                all_data.append(df)
                page += 1

                logger.info(
                    f"Loaded page {page-1}, total rows: {sum(len(d) for d in all_data)}")

            except Exception as e:
                logger.warning(f"Failed to load page {page}: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def post_data(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST data to API endpoint

        Args:
            endpoint: API endpoint
            data: Data to send

        Returns:
            Response data
        """
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.post(
                url,
                json=data,
                headers=self.headers,
                auth=self.auth,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"POST request failed: {e}")
            raise DataLoadError(f"Failed to POST data: {str(e)}")

    def close(self):
        """Close the session"""
        self.session.close()
        logger.info("API session closed")


class GraphQLLoader(DataLoader):
    """Load data from GraphQL APIs"""

    def __init__(self, endpoint: str, headers: Optional[Dict[str, str]] = None):
        """
        Initialize GraphQL loader

        Args:
            endpoint: GraphQL endpoint URL
            headers: HTTP headers
        """
        self.endpoint = endpoint
        self.headers = headers or {'Content-Type': 'application/json'}

    def validate(self, source: Any) -> bool:
        """Validate GraphQL endpoint"""
        try:
            # Try introspection query
            query = "{ __schema { types { name } } }"
            response = requests.post(
                self.endpoint,
                json={'query': query},
                headers=self.headers,
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def load(self, source: Any, variables: Optional[Dict] = None, **kwargs) -> pd.DataFrame:
        """
        Execute GraphQL query and load data

        Args:
            source: GraphQL query string
            variables: Query variables
            **kwargs: Additional parameters

        Returns:
            pd.DataFrame: Query results
        """
        try:
            payload = {
                'query': source,
                'variables': variables or {}
            }

            response = requests.post(
                self.endpoint,
                json=payload,
                headers=self.headers,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            if 'errors' in result:
                raise DataLoadError(f"GraphQL errors: {result['errors']}")

            # Extract data
            data = result.get('data', {})

            # Convert to DataFrame
            if isinstance(data, dict):
                # Get first key's value
                first_key = list(data.keys())[0]
                data = data[first_key]

            df = pd.DataFrame(data if isinstance(data, list) else [data])

            logger.info(f"Successfully loaded {len(df)} rows from GraphQL")
            return df

        except Exception as e:
            logger.error(f"GraphQL query failed: {e}")
            raise DataLoadError(f"Failed to execute GraphQL query: {str(e)}")


class WebScraperLoader(DataLoader):
    """Load data by scraping web pages"""

    def __init__(self):
        """Initialize web scraper loader"""
        self.session = requests.Session()

    def validate(self, source: Any) -> bool:
        """Validate URL is accessible"""
        try:
            response = self.session.head(source, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def load(self, source: Any, **kwargs) -> pd.DataFrame:
        """
        Scrape data from web page

        Args:
            source: URL to scrape
            **kwargs: Additional parameters

        Returns:
            pd.DataFrame: Scraped data
        """
        try:
            # Use pandas read_html for table extraction
            tables = pd.read_html(source)

            if not tables:
                raise DataLoadError("No tables found on page")

            # Return first table by default, or specified index
            table_index = kwargs.get('table_index', 0)
            df = tables[table_index]

            logger.info(f"Successfully scraped {len(df)} rows from {source}")
            return df

        except Exception as e:
            logger.error(f"Web scraping failed: {e}")
            raise DataLoadError(f"Failed to scrape data: {str(e)}")

    def close(self):
        """Close the session"""
        self.session.close()
