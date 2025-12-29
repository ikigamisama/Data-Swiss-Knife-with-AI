import requests
import json
import pandas as pd
import numpy as np

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from core.base import LLMAdapter, SingletonMeta


@dataclass
class OllamaConfig:
    """Configuration for Ollama client"""
    base_url: str = "http://localhost:11434"
    model: str = "gpt-oss-120b-cloud"
    timeout: int = 120
    temperature: float = 0.7
    max_tokens: int = 4096


class OllamaClient(LLMAdapter, metaclass=SingletonMeta):
    """Singleton Ollama client for LLM interactions"""

    def __init__(self, config: Optional[OllamaConfig] = None):
        if hasattr(self, '_initialized'):
            return

        self.config = config or OllamaConfig()
        self._initialized = True
        self._session = requests.Session()

    def is_available(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = self._session.get(
                f"{self.config.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = self._session.get(f"{self.config.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        except Exception as e:
            raise ConnectionError(f"Failed to list models: {e}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM"""
        url = f"{self.config.base_url}/api/generate"

        payload = {
            "model": kwargs.get("model", self.config.model),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            }
        }

        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"LLM request timed out after {self.config.timeout}s")
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}")

    def generate_streaming(self, prompt: str, **kwargs):
        """Generate response with streaming"""
        url = f"{self.config.base_url}/api/generate"

        payload = {
            "model": kwargs.get("model", self.config.model),
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
            }
        }

        try:
            response = self._session.post(
                url,
                json=payload,
                stream=True,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
                    if chunk.get("done", False):
                        break
        except Exception as e:
            raise RuntimeError(f"Streaming generation failed: {e}")

    def generate_code(self, instruction: str, context: str) -> str:
        """Generate Python code from natural language instruction"""
        prompt = f"""You are an expert Python data analyst. Generate clean, efficient Python code.

Context:
{context}

Instruction: {instruction}

Requirements:
- Use pandas, numpy, matplotlib as needed
- Code should be production-ready
- Include error handling
- Add brief comments
- Return only the code in a ```python``` block

Generate the code:"""

        response = self.generate(prompt, temperature=0.3)

        # Extract code from markdown blocks
        code = self._extract_code_block(response)
        return code

    def nl_to_pandas(self, query: str, df_info: Dict[str, Any]) -> str:
        """Convert natural language to pandas code"""
        columns_info = ", ".join([
            f"{col}({dtype})"
            for col, dtype in df_info.get("dtypes", {}).items()
        ])

        prompt = f"""Convert this natural language query to pandas code.

DataFrame info:
- Shape: {df_info.get('shape', 'Unknown')}
- Columns: {columns_info}
- Sample data available in variable: df

Query: {query}

Return only the pandas code (no explanations):"""

        code = self.generate(prompt, temperature=0.2, max_tokens=500)
        return self._extract_code_block(code)

    def explain_data(self, summary: Dict[str, Any]) -> str:
        """Generate natural language explanation of data"""
        
        prompt = f"""Analyze this dataset summary and provide insights:

{json.dumps(summary, indent=2, default=str)}

Provide:
1. Key observations (3-5 points)
2. Data quality issues if any
3. Recommended next steps for analysis

Keep it concise and actionable:"""

        return self.generate(prompt, temperature=0.8)

    def suggest_visualizations(self, df_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """Suggest appropriate visualizations for the dataset"""
        prompt = f"""Given this dataset information, suggest 5 most appropriate visualizations:

Dataset info:
- Shape: {df_info.get('shape')}
- Numeric columns: {df_info.get('numeric_columns', [])}
- Categorical columns: {df_info.get('categorical_columns', [])}
- DateTime columns: {df_info.get('datetime_columns', [])}

Return a JSON array with format:
[
  {{"type": "histogram", "reason": "...", "columns": ["col1"]}},
  ...
]

Respond with only the JSON array:"""

        response = self.generate(prompt, temperature=0.5)
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return []
        except:
            return []

    def optimize_query(self, query: str, dialect: str = "pandas") -> str:
        """Optimize a data query for performance"""
        prompt = f"""Optimize this {dialect} query for better performance:

Query:
{query}

Provide:
1. Optimized version
2. Explanation of improvements
3. Estimated performance gain

Format as:
OPTIMIZED CODE:
```python
...
```

EXPLANATION:
..."""

        return self.generate(prompt, temperature=0.3)

    def _extract_code_block(self, text: str) -> str:
        """Extract code from markdown code blocks"""
        import re

        # Try to find python code block
        pattern = r'```python\n(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try generic code block
        pattern = r'```\n(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Return as-is if no code block found
        return text.strip()
    

    def _json_safe(self, obj):


        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._json_safe(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._json_safe(v) for v in obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        elif isinstance(obj, (pd.api.extensions.ExtensionDtype,)):
            return str(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj



# Convenience function for quick access
def get_ollama_client(model: str = "gpt-oss-120b-cloud") -> OllamaClient:
    """Get or create singleton Ollama client"""
    config = OllamaConfig(model=model)
    return OllamaClient(config)
