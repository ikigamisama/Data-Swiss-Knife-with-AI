"""
ML Models Page - AutoML and Model Training
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report
)


st.set_page_config(page_title="ML Models", page_icon="📉", layout="wide")

st.title("📉 Machine Learning Models")
st.markdown("AutoML and model training interface")

# Check if data is loaded
if st.session_state.get('data') is None:
    st.warning("⚠️ No data loaded. Please load data from the main page.")
    st.stop()

df = st.session_state.data

# Initialize model state
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = []

# Sidebar
with st.sidebar:
    st.header("🎯 Model Configuration")

    problem_type = st.selectbox(
        "Problem Type",
        ["Classification", "Regression", "Clustering", "Time Series"]
    )

    st.markdown("---")

    # Feature selection
    st.subheader("📊 Features")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=['object', 'category']).columns.tolist()

    if problem_type in ["Classification", "Regression"]:
        target_col = st.selectbox("Target Variable", df.columns.tolist())

        available_features = [col for col in df.columns if col != target_col]
        feature_cols = st.multiselect(
            "Feature Variables",
            available_features,
            default=[col for col in numeric_cols if col != target_col][:5]
        )
    else:
        feature_cols = st.multiselect(
            "Feature Variables",
            numeric_cols,
            default=numeric_cols[:5]
        )

    st.markdown("---")

    # Model selection
    st.subheader("🤖 Model Selection")

    if problem_type == "Classification":
        model_type = st.selectbox(
            "Algorithm",
            ["Logistic Regression", "Random Forest",
                "XGBoost", "SVM", "KNN", "Naive Bayes"]
        )
    elif problem_type == "Regression":
        model_type = st.selectbox(
            "Algorithm",
            ["Linear Regression", "Random Forest",
                "XGBoost", "SVR", "KNN", "Ridge", "Lasso"]
        )
    elif problem_type == "Clustering":
        model_type = st.selectbox(
            "Algorithm",
            ["K-Means", "DBSCAN", "Hierarchical", "Gaussian Mixture"]
        )
    else:
        model_type = st.selectbox(
            "Algorithm",
            ["ARIMA", "Prophet", "LSTM", "Exponential Smoothing"]
        )

    st.markdown("---")

    # Training parameters
    st.subheader("⚙️ Parameters")

    if problem_type in ["Classification", "Regression"]:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random State", 0, 999, 42)

    if problem_type == "Clustering":
        if model_type == "K-Means":
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        elif model_type == "DBSCAN":
            eps = st.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("Min Samples", 2, 10, 5)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔨 Train Model", "📊 Evaluation", "🔮 Predictions", "💾 Model Management"])

# Tab 1: Train Model
with tab1:
    st.subheader("🔨 Model Training")

    if not feature_cols or (problem_type in ["Classification", "Regression"] and not target_col):
        st.warning("⚠️ Please select features and target variable in the sidebar")
    else:
        # Display data info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(feature_cols))
        with col3:
            if problem_type in ["Classification", "Regression"]:
                st.metric("Target", target_col)

        # Data preview
        st.markdown("---")
        st.write("### Data Preview")

        if problem_type in ["Classification", "Regression"]:
            preview_df = df[feature_cols + [target_col]].head(10)
        else:
            preview_df = df[feature_cols].head(10)

        st.dataframe(preview_df, width='stretch')

        # Check for missing values
        missing_counts = df[feature_cols].isnull().sum()
        if missing_counts.sum() > 0:
            st.warning(
                "⚠️ Missing values detected. Consider cleaning data first.")
            st.write(missing_counts[missing_counts > 0])

        # Train button
        st.markdown("---")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            train_button = st.button(
                "🚀 Train Model", type="primary", width='stretch')

        with col2:
            auto_tune = st.checkbox("Auto-tune", value=False)

        with col3:
            cross_val = st.checkbox("Cross-validation", value=True)

        # Training logic
        if train_button:
            with st.spinner("🔄 Training model..."):
                try:
                    # Prepare data
                    if problem_type in ["Classification", "Regression"]:
                        X = df[feature_cols].dropna()
                        y = df.loc[X.index, target_col]

                        # Handle categorical features
                        X_processed = X.copy()
                        for col in categorical_cols:
                            if col in X_processed.columns:
                                le = LabelEncoder()
                                X_processed[col] = le.fit_transform(
                                    X_processed[col].astype(str))

                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_processed, y, test_size=test_size, random_state=random_state
                        )

                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        # Train model
                        if problem_type == "Classification":
                            if model_type == "Logistic Regression":
                                from sklearn.linear_model import LogisticRegression
                                model = LogisticRegression(
                                    random_state=random_state, max_iter=1000)
                            elif model_type == "Random Forest":
                                from sklearn.ensemble import RandomForestClassifier
                                model = RandomForestClassifier(
                                    random_state=random_state, n_estimators=100)
                            elif model_type == "KNN":
                                from sklearn.neighbors import KNeighborsClassifier
                                model = KNeighborsClassifier(n_neighbors=5)
                            elif model_type == "Naive Bayes":
                                from sklearn.naive_bayes import GaussianNB
                                model = GaussianNB()
                            else:
                                st.error(f"{model_type} not yet implemented")
                                st.stop()

                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)

                            # Calculate metrics
                            metrics = {
                                'accuracy': accuracy_score(y_test, y_pred),
                                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            }

                            # Confusion matrix
                            cm = confusion_matrix(y_test, y_pred)

                        else:  # Regression
                            if model_type == "Linear Regression":
                                from sklearn.linear_model import LinearRegression
                                model = LinearRegression()
                            elif model_type == "Random Forest":
                                from sklearn.ensemble import RandomForestRegressor
                                model = RandomForestRegressor(
                                    random_state=random_state, n_estimators=100)
                            elif model_type == "Ridge":
                                from sklearn.linear_model import Ridge
                                model = Ridge(random_state=random_state)
                            elif model_type == "Lasso":
                                from sklearn.linear_model import Lasso
                                model = Lasso(random_state=random_state)
                            elif model_type == "KNN":
                                from sklearn.neighbors import KNeighborsRegressor
                                model = KNeighborsRegressor(n_neighbors=5)
                            else:
                                st.error(f"{model_type} not yet implemented")
                                st.stop()

                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)

                            # Calculate metrics
                            metrics = {
                                'mse': mean_squared_error(y_test, y_pred),
                                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                                'mae': mean_absolute_error(y_test, y_pred),
                                'r2': r2_score(y_test, y_pred)
                            }

                            cm = None

                        # Save model
                        model_info = {
                            'name': f"{model_type}_{len(st.session_state.trained_models) + 1}",
                            'type': problem_type,
                            'algorithm': model_type,
                            'model': model,
                            'scaler': scaler,
                            'features': feature_cols,
                            'target': target_col if problem_type in ["Classification", "Regression"] else None,
                            'metrics': metrics,
                            'confusion_matrix': cm,
                            'X_test': X_test_scaled,
                            'y_test': y_test,
                            'y_pred': y_pred
                        }

                        st.session_state.trained_models.append(model_info)

                        st.success("✅ Model trained successfully!")
                        st.balloons()

                        # Display metrics
                        st.markdown("---")
                        st.write("### 📊 Training Results")

                        cols = st.columns(len(metrics))
                        for idx, (metric_name, metric_value) in enumerate(metrics.items()):
                            with cols[idx]:
                                st.metric(metric_name.upper(),
                                          f"{metric_value:.4f}")

                    elif problem_type == "Clustering":
                        X = df[feature_cols].dropna()

                        # Scale features
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)

                        # Train model
                        if model_type == "K-Means":
                            from sklearn.cluster import KMeans
                            model = KMeans(
                                n_clusters=n_clusters, random_state=42)
                            labels = model.fit_predict(X_scaled)
                        else:
                            st.error(f"{model_type} not yet implemented")
                            st.stop()

                        # Save model
                        model_info = {
                            'name': f"{model_type}_{len(st.session_state.trained_models) + 1}",
                            'type': problem_type,
                            'algorithm': model_type,
                            'model': model,
                            'scaler': scaler,
                            'features': feature_cols,
                            'labels': labels,
                            'X': X_scaled
                        }

                        st.session_state.trained_models.append(model_info)
                        st.success("✅ Model trained successfully!")

                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    st.code(str(e))

# Tab 2: Evaluation
with tab2:
    st.subheader("📊 Model Evaluation")

    if not st.session_state.trained_models:
        st.info("👆 Train a model first to see evaluation metrics")
    else:
        model_names = [m['name'] for m in st.session_state.trained_models]
        selected_model_name = st.selectbox("Select Model", model_names)

        model_info = next(
            m for m in st.session_state.trained_models if m['name'] == selected_model_name)

        st.markdown("---")

        if model_info['type'] == "Classification":
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            metrics = model_info['metrics']

            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1']:.4f}")

            # Confusion Matrix
            st.markdown("---")
            st.write("### Confusion Matrix")

            cm = model_info['confusion_matrix']
            fig = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, width='stretch')

        elif model_info['type'] == "Regression":
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            metrics = model_info['metrics']

            with col1:
                st.metric("MSE", f"{metrics['mse']:.4f}")
            with col2:
                st.metric("RMSE", f"{metrics['rmse']:.4f}")
            with col3:
                st.metric("MAE", f"{metrics['mae']:.4f}")
            with col4:
                st.metric("R² Score", f"{metrics['r2']:.4f}")

            # Actual vs Predicted
            st.markdown("---")
            st.write("### Actual vs Predicted")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=model_info['y_test'],
                y=model_info['y_pred'],
                mode='markers',
                name='Predictions'
            ))

            # Perfect prediction line
            min_val = min(model_info['y_test'].min(),
                          model_info['y_pred'].min())
            max_val = max(model_info['y_test'].max(),
                          model_info['y_pred'].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))

            fig.update_layout(
                xaxis_title='Actual',
                yaxis_title='Predicted',
                title='Actual vs Predicted Values'
            )
            st.plotly_chart(fig, width='stretch')

            # Residuals
            st.write("### Residuals Plot")
            residuals = model_info['y_test'] - model_info['y_pred']

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=model_info['y_pred'],
                y=residuals,
                mode='markers'
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(
                xaxis_title='Predicted Values',
                yaxis_title='Residuals',
                title='Residual Plot'
            )
            st.plotly_chart(fig, width='stretch')

        elif model_info['type'] == "Clustering":
            # Cluster visualization (2D PCA)
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(model_info['X'])

            cluster_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Cluster': model_info['labels']
            })

            fig = px.scatter(
                cluster_df,
                x='PC1',
                y='PC2',
                color='Cluster',
                title='Cluster Visualization (PCA)'
            )
            st.plotly_chart(fig, width='stretch')

# Tab 3: Predictions
with tab3:
    st.subheader("🔮 Make Predictions")

    if not st.session_state.trained_models:
        st.info("👆 Train a model first to make predictions")
    else:
        model_names = [m['name'] for m in st.session_state.trained_models]
        selected_model_name = st.selectbox(
            "Select Model", model_names, key="pred_model")

        model_info = next(
            m for m in st.session_state.trained_models if m['name'] == selected_model_name)

        st.markdown("---")

        prediction_mode = st.radio("Prediction Mode", ["Single", "Batch"])

        if prediction_mode == "Single":
            st.write("### Enter Feature Values")

            input_data = {}
            cols = st.columns(3)

            for idx, feature in enumerate(model_info['features']):
                with cols[idx % 3]:
                    input_data[feature] = st.number_input(
                        f"{feature}", value=0.0)

            if st.button("🔮 Predict", type="primary"):
                input_df = pd.DataFrame([input_data])
                input_scaled = model_info['scaler'].transform(input_df)
                prediction = model_info['model'].predict(input_scaled)[0]

                st.success(f"### Prediction: {prediction:.4f}")

        else:
            st.write("### Upload Prediction Data")
            st.info("Upload a CSV file with the same features as training data")

            pred_file = st.file_uploader("Choose file", type=['csv'])

            if pred_file:
                pred_df = pd.read_csv(pred_file)
                st.dataframe(pred_df.head())

                if st.button("🔮 Predict All"):
                    pred_scaled = model_info['scaler'].transform(
                        pred_df[model_info['features']])
                    predictions = model_info['model'].predict(pred_scaled)

                    pred_df['Prediction'] = predictions
                    st.dataframe(pred_df)

                    csv = pred_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Predictions",
                        csv,
                        "predictions.csv",
                        "text/csv"
                    )

# Tab 4: Model Management
with tab4:
    st.subheader("💾 Trained Models")

    if not st.session_state.trained_models:
        st.info("No trained models yet")
    else:
        for idx, model in enumerate(st.session_state.trained_models):
            with st.expander(f"📦 {model['name']}", expanded=(idx == 0)):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"**Type:** {model['type']}")
                    st.write(f"**Algorithm:** {model['algorithm']}")
                    st.write(f"**Features:** {', '.join(model['features'])}")
                    if model.get('target'):
                        st.write(f"**Target:** {model['target']}")

                with col2:
                    if st.button("🗑️ Delete", key=f"del_{idx}"):
                        st.session_state.trained_models.pop(idx)
                        st.rerun()

                if model['metrics']:
                    st.write("**Metrics:**")
                    metric_cols = st.columns(len(model['metrics']))
                    for idx2, (k, v) in enumerate(model['metrics'].items()):
                        with metric_cols[idx2]:
                            st.metric(k.upper(), f"{v:.4f}")
