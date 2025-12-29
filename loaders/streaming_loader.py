"""
Streaming Data Loader - Real-time Data Streaming
"""
import pandas as pd
import numpy as np
from typing import Any, Callable, Optional, Generator
from queue import Queue
from threading import Thread
import time
import logging

logger = logging.getLogger(__name__)


class StreamingLoader:
    """Load data from real-time streams"""
    
    def __init__(self, buffer_size: int = 1000):
        """
        Initialize streaming loader
        
        Args:
            buffer_size: Size of internal buffer
        """
        self.buffer_size = buffer_size
        self.buffer = Queue(maxsize=buffer_size)
        self.is_running = False
        self.thread = None
    
    def start_stream(self, source: Any, callback: Optional[Callable] = None):
        """
        Start streaming data
        
        Args:
            source: Data source (WebSocket, Kafka, etc.)
            callback: Function to call with each data chunk
        """
        self.is_running = True
        self.thread = Thread(target=self._stream_data, args=(source, callback))
        self.thread.start()
        logger.info("Started streaming data")
    
    def stop_stream(self):
        """Stop streaming"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        logger.info("Stopped streaming data")
    
    def _stream_data(self, source: Any, callback: Optional[Callable]):
        """Internal streaming loop"""
        while self.is_running:
            try:
                # Get data from source
                data = self._fetch_data(source)
                
                if data is not None:
                    self.buffer.put(data)
                    
                    if callback:
                        callback(data)
                
                time.sleep(0.1)  # Prevent busy waiting
                
            except Exception as e:
                logger.error(f"Streaming error: {e}")
    
    def _fetch_data(self, source: Any) -> Optional[pd.DataFrame]:
        """Fetch data from source"""
        # Implementation depends on source type
        return None
    
    def get_buffered_data(self) -> pd.DataFrame:
        """Get all buffered data"""
        data_list = []
        while not self.buffer.empty():
            data_list.append(self.buffer.get())
        
        if data_list:
            return pd.concat(data_list, ignore_index=True)
        return pd.DataFrame()
    
    def stream_generator(self, source: Any) -> Generator[pd.DataFrame, None, None]:
        """
        Create a generator for streaming data
        
        Args:
            source: Data source
            
        Yields:
            DataFrame chunks
        """
        while self.is_running:
            data = self._fetch_data(source)
            if data is not None:
                yield data
            time.sleep(0.1)


class KafkaStreamLoader(StreamingLoader):
    """Load data from Kafka streams"""
    
    def __init__(self, bootstrap_servers: str, topic: str, **kwargs):
        """
        Initialize Kafka stream loader
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            topic: Kafka topic to consume from
            **kwargs: Additional Kafka consumer parameters
        """
        super().__init__()
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer = None
    
    def _fetch_data(self, source: Any) -> Optional[pd.DataFrame]:
        """Fetch data from Kafka"""
        try:
            from kafka import KafkaConsumer
            import json
            
            if self.consumer is None:
                self.consumer = KafkaConsumer(
                    self.topic,
                    bootstrap_servers=self.bootstrap_servers,
                    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
                )
            
            message = next(self.consumer)
            data = message.value
            
            return pd.DataFrame([data])
            
        except ImportError:
            logger.error("kafka-python not installed")
            return None
        except Exception as e:
            logger.error(f"Kafka fetch error: {e}")
            return None


class WebSocketStreamLoader(StreamingLoader):
    """Load data from WebSocket streams"""
    
    def __init__(self, ws_url: str):
        """
        Initialize WebSocket stream loader
        
        Args:
            ws_url: WebSocket URL
        """
        super().__init__()
        self.ws_url = ws_url
        self.ws = None
    
    def _fetch_data(self, source: Any) -> Optional[pd.DataFrame]:
        """Fetch data from WebSocket"""
        try:
            import websocket
            import json
            
            if self.ws is None:
                self.ws = websocket.create_connection(self.ws_url)
            
            message = self.ws.recv()
            data = json.loads(message)
            
            return pd.DataFrame([data])
            
        except ImportError:
            logger.error("websocket-client not installed")
            return None
        except Exception as e:
            logger.error(f"WebSocket fetch error: {e}")
            return None

