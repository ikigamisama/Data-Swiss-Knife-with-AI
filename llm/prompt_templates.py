from abc import ABC, abstractmethod
from typing import Dict, Any, List
import json


class PromptTemplate(ABC):
    """Base class for prompt templates"""
    
    def __init__(self):
        self.system_message = self._get_system_message()
    
    @abstractmethod
    def _get_system_message(self) -> str:
        """Define the system message for this template"""
        pass
    
    @abstractmethod
    def format(self, **kwargs) -> str:
        """Format the prompt with provided arguments"""
        pass
    
    def build_prompt(self, **kwargs) -> str:
        """Build complete prompt with system message"""
        user_message = self.format(**kwargs)
        return f"{self.system_message}\n\n{user_message}"


class CodeGenerationTemplate(PromptTemplate):
    """Template for generating Python code"""
    
    def _get_system_message(self) -> str:
        return """You are an expert Python data analyst and programmer.
Generate clean, efficient, and well-documented Python code.
Use pandas, numpy, matplotlib, and plotly as needed.
Always include error handling and type hints where appropriate.
Return ONLY the code without explanations, wrapped in ```python``` blocks."""
    
    def format(self, instruction: str, context: Dict[str, Any]) -> str:
        """Format code generation prompt"""
        
        df_info = self._format_dataframe_info(context)
        
        prompt = f"""Given this dataset information:
{df_info}

Task: {instruction}

Requirements:
- Write production-ready Python code
- Use pandas operations efficiently
- Handle edge cases and errors
- Include brief inline comments
- Assume the DataFrame is available as 'df'

Generate the code:"""
        
        return prompt
    
    def _format_dataframe_info(self, context: Dict[str, Any]) -> str:
        """Format DataFrame information for prompt"""
        info_parts = []
        
        if 'shape' in context:
            info_parts.append(f"Shape: {context['shape']}")
        
        if 'columns' in context:
            info_parts.append(f"Columns: {', '.join(context['columns'][:20])}")
        
        if 'dtypes' in context:
            dtypes_str = '\n'.join([f"  - {col}: {dtype}" for col, dtype in list(context['dtypes'].items())[:10]])
            info_parts.append(f"Data Types:\n{dtypes_str}")
        
        if 'numeric_columns' in context:
            info_parts.append(f"Numeric Columns: {', '.join(context['numeric_columns'][:10])}")
        
        if 'categorical_columns' in context:
            info_parts.append(f"Categorical Columns: {', '.join(context['categorical_columns'][:10])}")
        
        return '\n'.join(info_parts)


class DataExplanationTemplate(PromptTemplate):
    """Template for explaining data insights"""
    
    def _get_system_message(self) -> str:
        return """You are a data analyst expert who explains complex data insights in clear, actionable language.
Focus on:
1. Key observations and patterns
2. Potential data quality issues
3. Actionable recommendations
4. Business implications

Be concise but thorough. Use bullet points for clarity."""
    
    def format(self, summary: Dict[str, Any], focus: str = 'general') -> str:
        """Format data explanation prompt"""
        
        summary_json = json.dumps(summary, indent=2, default=str)
        
        if focus == 'general':
            task = "Provide a comprehensive analysis covering all aspects"
        elif focus == 'quality':
            task = "Focus on data quality issues and recommendations for improvement"
        elif focus == 'insights':
            task = "Focus on interesting patterns and actionable insights"
        elif focus == 'anomalies':
            task = "Focus on unusual patterns, outliers, and potential anomalies"
        else:
            task = focus
        
        prompt = f"""Analyze this dataset summary:

{summary_json}

Task: {task}

Provide your analysis in the following format:

## Key Observations
[3-5 most important findings]

## Data Quality Assessment
[Quality issues if any, with severity]

## Recommendations
[Actionable next steps]

## Business Implications
[How these findings might impact decision-making]"""
        
        return prompt


class QueryTranslationTemplate(PromptTemplate):
    """Template for natural language to code translation"""
    
    def _get_system_message(self) -> str:
        return """You are an expert at translating natural language questions into pandas code.
Generate concise, efficient pandas code that directly answers the question.
Return ONLY the code without explanations."""
    
    def format(self, query: str, df_info: Dict[str, Any]) -> str:
        """Format query translation prompt"""
        
        columns_info = ", ".join([
            f"{col} ({dtype})" 
            for col, dtype in list(df_info.get('dtypes', {}).items())[:20]
        ])
        
        prompt = f"""Dataset Information:
- Shape: {df_info.get('shape', 'Unknown')}
- Columns: {columns_info}

Natural Language Query: "{query}"

Translate this to pandas code. The DataFrame is available as 'df'.
Store the result in a variable called 'result'.

Important:
- Use efficient pandas operations
- Handle potential errors
- Keep it simple and readable

Code:"""
        
        return prompt


class VisualizationSuggestionTemplate(PromptTemplate):
    """Template for suggesting visualizations"""
    
    def _get_system_message(self) -> str:
        return """You are a data visualization expert.
Suggest the most appropriate visualizations for the given dataset.
Consider the data types, relationships, and analytical goals.
Return suggestions in JSON format."""
    
    def format(self, df_info: Dict[str, Any], goal: str = 'exploration') -> str:
        """Format visualization suggestion prompt"""
        
        prompt = f"""Dataset Information:
- Shape: {df_info.get('shape')}
- Numeric Columns: {len(df_info.get('numeric_columns', []))}
- Categorical Columns: {len(df_info.get('categorical_columns', []))}
- DateTime Columns: {len(df_info.get('datetime_columns', []))}

Analysis Goal: {goal}

Suggest 5 most appropriate visualizations for this dataset.

Return a JSON array with this format:
[
  {{
    "type": "scatter",
    "title": "Relationship between X and Y",
    "reason": "Useful for identifying correlations",
    "columns": ["col1", "col2"],
    "library": "plotly"
  }},
  ...
]

Focus on:
- Practical insights
- Data types compatibility
- Ease of interpretation

JSON Response:"""
        
        return prompt


class OptimizationTemplate(PromptTemplate):
    """Template for code optimization suggestions"""
    
    def _get_system_message(self) -> str:
        return """You are a performance optimization expert for pandas and data processing.
Analyze code and suggest optimizations for speed and memory efficiency.
Provide concrete, actionable improvements."""
    
    def format(self, code: str, context: str = '') -> str:
        """Format optimization prompt"""
        
        prompt = f"""Analyze this pandas code for optimization opportunities:

```python
{code}
```

Context: {context}

Provide:
1. **Performance Issues**: Identify bottlenecks
2. **Optimized Code**: Provide improved version
3. **Explanation**: Why these changes improve performance
4. **Expected Impact**: Estimated performance gain

Format your response as:

## Performance Analysis
[List of issues found]

## Optimized Code
```python
[Improved code here]
```

## Explanation
[Why these optimizations work]

## Expected Impact
[Performance improvements]"""
        
        return prompt


class SQLGenerationTemplate(PromptTemplate):
    """Template for generating SQL queries"""
    
    def _get_system_message(self) -> str:
        return """You are a SQL expert. Generate optimized, readable SQL queries.
Use standard SQL syntax compatible with PostgreSQL, MySQL, and SQLite.
Include comments for complex logic."""
    
    def format(self, request: str, schema: Dict[str, Any]) -> str:
        """Format SQL generation prompt"""
        
        tables_info = []
        for table, columns in schema.items():
            cols = ', '.join(columns) if isinstance(columns, list) else columns
            tables_info.append(f"  {table} ({cols})")
        
        schema_str = '\n'.join(tables_info)
        
        prompt = f"""Database Schema:
{schema_str}

Request: {request}

Generate an optimized SQL query that:
- Is readable and well-formatted
- Uses appropriate JOINs
- Includes WHERE clauses for filtering
- Has proper indexing hints if needed
- Includes comments for complex logic

SQL Query:"""
        
        return prompt


class FeatureEngineeringTemplate(PromptTemplate):
    """Template for feature engineering suggestions"""
    
    def _get_system_message(self) -> str:
        return """You are a feature engineering expert.
Suggest creative and effective features for machine learning based on the dataset.
Consider domain knowledge and statistical principles."""
    
    def format(self, df_info: Dict[str, Any], target: str, problem_type: str) -> str:
        """Format feature engineering prompt"""
        
        prompt = f"""Dataset Information:
- Columns: {', '.join(df_info.get('columns', [])[:20])}
- Numeric: {len(df_info.get('numeric_columns', []))}
- Categorical: {len(df_info.get('categorical_columns', []))}
- Target Variable: {target}
- Problem Type: {problem_type}

Suggest 5-10 engineered features that could improve model performance.

For each feature, provide:
1. Feature Name
2. Derivation Logic (in pandas code)
3. Rationale
4. Expected Impact

Format as:

## Suggested Features

### 1. [Feature Name]
```python
# Pandas code to create this feature
```
**Rationale**: Why this feature is useful
**Expected Impact**: How it might help the model

[Continue for each feature...]"""
        
        return prompt


class DocumentationTemplate(PromptTemplate):
    """Template for generating code documentation"""
    
    def _get_system_message(self) -> str:
        return """You are a technical documentation expert.
Generate clear, comprehensive documentation for data analysis code.
Include docstrings, type hints, and usage examples."""
    
    def format(self, code: str, purpose: str = '') -> str:
        """Format documentation prompt"""
        
        prompt = f"""Add comprehensive documentation to this code:

```python
{code}
```

Purpose: {purpose}

Provide:
1. Function docstrings (Google style)
2. Inline comments for complex logic
3. Type hints
4. Usage examples
5. Edge cases and error handling notes

Documented Code:"""
        
        return prompt


class ErrorDiagnosisTemplate(PromptTemplate):
    """Template for diagnosing and fixing errors"""
    
    def _get_system_message(self) -> str:
        return """You are a debugging expert for data analysis code.
Diagnose errors and provide clear solutions with corrected code."""
    
    def format(self, code: str, error: str, context: str = '') -> str:
        """Format error diagnosis prompt"""
        
        prompt = f"""Analyze this error and provide a fix:

**Code:**
```python
{code}
```

**Error:**
```
{error}
```

**Context:** {context}

Provide:
1. **Error Explanation**: What caused the error
2. **Root Cause**: Why it happened
3. **Fixed Code**: Corrected version
4. **Prevention**: How to avoid similar errors

Format your response clearly with code blocks."""
        
        return prompt


# Factory for getting templates
class PromptTemplateFactory:
    """Factory for creating prompt templates"""
    
    _templates = {
        'code_generation': CodeGenerationTemplate,
        'data_explanation': DataExplanationTemplate,
        'query_translation': QueryTranslationTemplate,
        'visualization': VisualizationSuggestionTemplate,
        'optimization': OptimizationTemplate,
        'sql_generation': SQLGenerationTemplate,
        'feature_engineering': FeatureEngineeringTemplate,
        'documentation': DocumentationTemplate,
        'error_diagnosis': ErrorDiagnosisTemplate
    }
    
    @classmethod
    def get_template(cls, template_type: str) -> PromptTemplate:
        """Get a prompt template by type"""
        if template_type not in cls._templates:
            raise ValueError(f"Unknown template type: {template_type}")
        
        return cls._templates[template_type]()
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """List all available template types"""
        return list(cls._templates.keys())