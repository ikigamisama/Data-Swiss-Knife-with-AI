import streamlit as st
from typing import Dict, Any, Optional


def render_sidebar(
    data_loaded: bool = False,
    llm_available: bool = False
) -> Dict[str, Any]:
    """
    Render main sidebar with navigation and settings

    Args:
        data_loaded: Whether data is loaded
        llm_available: Whether LLM is available

    Returns:
        Dictionary with user selections
    """
    with st.sidebar:
        st.title("🔪 Swiss Knife")

        # Logo/Image
        st.image("https://via.placeholder.com/150", width=150)

        st.markdown("---")

        # Status indicators
        st.subheader("📊 Status")

        col1, col2 = st.columns(2)
        with col1:
            if data_loaded:
                st.success("✅ Data")
            else:
                st.error("❌ Data")

        with col2:
            if llm_available:
                st.success("✅ AI")
            else:
                st.error("❌ AI")

        st.markdown("---")

        # Quick actions
        st.subheader("⚡ Quick Actions")

        actions = {}

        if st.button("🔄 Refresh Data", width='stretch'):
            actions['refresh'] = True

        if st.button("📥 Export Results", width='stretch'):
            actions['export'] = True

        if st.button("🗑️ Clear Session", width='stretch'):
            actions['clear'] = True

        st.markdown("---")

        # Settings
        st.subheader("⚙️ Settings")

        settings = {}
        settings['theme'] = st.selectbox(
            "Theme",
            ["Light", "Dark", "Auto"],
            key="theme_selector"
        )

        settings['auto_save'] = st.checkbox("Auto-save", value=True)
        settings['show_warnings'] = st.checkbox("Show warnings", value=True)

        st.markdown("---")

        # Help
        with st.expander("❓ Help & Support"):
            st.markdown("""
            **Quick Links:**
            - [Documentation](https://docs.example.com)
            - [Tutorials](https://docs.example.com/tutorials)
            - [GitHub](https://github.com/example/repo)
            - [Discord](https://discord.gg/example)
            
            **Keyboard Shortcuts:**
            - `Ctrl+S`: Save
            - `Ctrl+R`: Refresh
            - `Ctrl+K`: Command palette
            """)

        return {**actions, **settings}
