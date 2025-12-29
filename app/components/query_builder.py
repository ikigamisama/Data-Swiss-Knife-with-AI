import streamlit as st
import pandas as pd
from typing import Optional, List, Dict


class QueryBuilder:
    """Visual query builder for data filtering"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize query builder

        Args:
            df: DataFrame to query
        """
        self.df = df
        self.conditions: List[Dict] = []

    def render(self, key: str = "query_builder") -> Optional[pd.DataFrame]:
        """
        Render query builder

        Args:
            key: Unique key for widget

        Returns:
            Filtered DataFrame
        """
        st.subheader("🔍 Query Builder")

        # Add condition button
        if st.button("➕ Add Condition", key=f"{key}_add"):
            self.conditions.append({})

        # Render each condition
        for idx, condition in enumerate(self.conditions):
            self._render_condition(idx, condition, key)

        # Apply button
        if self.conditions and st.button("▶️ Apply Query", type="primary", key=f"{key}_apply"):
            return self._apply_conditions()

        return None

    def _render_condition(self, idx: int, condition: Dict, key: str):
        """Render a single condition"""
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

            with col1:
                column = st.selectbox(
                    "Column",
                    self.df.columns.tolist(),
                    key=f"{key}_col_{idx}"
                )
                condition['column'] = column

            with col2:
                # Determine operators based on column type
                if pd.api.types.is_numeric_dtype(self.df[column]):
                    operators = ['=', '!=', '>', '<', '>=', '<=']
                else:
                    operators = ['=', '!=', 'contains',
                                 'startswith', 'endswith']

                operator = st.selectbox(
                    "Operator",
                    operators,
                    key=f"{key}_op_{idx}"
                )
                condition['operator'] = operator

            with col3:
                value = st.text_input(
                    "Value",
                    key=f"{key}_val_{idx}"
                )
                condition['value'] = value

            with col4:
                if st.button("🗑️", key=f"{key}_del_{idx}"):
                    self.conditions.pop(idx)
                    st.rerun()

    def _apply_conditions(self) -> pd.DataFrame:
        """Apply all conditions and return filtered DataFrame"""
        result = self.df.copy()

        for condition in self.conditions:
            column = condition.get('column')
            operator = condition.get('operator')
            value = condition.get('value')

            if not all([column, operator, value]):
                continue

            try:
                if operator == '=':
                    result = result[result[column] == value]
                elif operator == '!=':
                    result = result[result[column] != value]
                elif operator == '>':
                    result = result[result[column] > float(value)]
                elif operator == '<':
                    result = result[result[column] < float(value)]
                elif operator == '>=':
                    result = result[result[column] >= float(value)]
                elif operator == '<=':
                    result = result[result[column] <= float(value)]
                elif operator == 'contains':
                    result = result[result[column].astype(
                        str).str.contains(value, na=False)]
                elif operator == 'startswith':
                    result = result[result[column].astype(
                        str).str.startswith(value, na=False)]
                elif operator == 'endswith':
                    result = result[result[column].astype(
                        str).str.endswith(value, na=False)]
            except Exception as e:
                st.error(f"Error applying condition: {e}")

        return result
