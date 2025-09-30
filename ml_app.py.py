import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification, make_blobs, fetch_california_housing
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# -------------------- Utilities -------------------- #
def load_demo_dataset(name, task="classification"):
    if task == "classification":
        if name == "Iris":
            data = load_iris(as_frame=True)
            return data.frame
        elif name == "Wine":
            data = load_wine(as_frame=True)
            return data.frame
        elif name == "Breast Cancer":
            data = load_breast_cancer(as_frame=True)
            return data.frame
        elif name == "Synthetic Classification":
            X, y = make_classification(n_samples=500, n_features=6, n_informative=4,
                                       n_classes=3, random_state=42)
            df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
            df["Target"] = y
            return df
    elif task == "regression":
        if name == "California Housing":
            data = fetch_california_housing(as_frame=True)
            return data.frame
    elif task == "clustering":
        if name == "Synthetic Blobs":
            X, y = make_blobs(n_samples=500, n_features=4, centers=3, random_state=42)
            return pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
    return pd.DataFrame()

# -------------------- Streamlit App -------------------- #
st.set_page_config(page_title="ML Analysis App", page_icon="📊", layout="wide")

st.title("📊 Data Analysis and Machine Learning App")
st.write("""
This app allows you to perform:
- 📂 Data Upload  
- 🔍 Exploratory Data Analysis (EDA)  
- 📈 Regression  
- 🔢 Classification  
- 📉 Clustering  
- 🚀 Model Deployment (reuse trained model or upload one)
""")

st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a task:",
    ["Data Upload", "Exploratory Data Analysis", "Regression", "Classification", "Clustering", "Model Deployment"]
)

# -------------------- DATA UPLOAD -------------------- #
if options == "Data Upload":
    st.header("📂 Upload your dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    demo_choice = st.selectbox("Or load a demo dataset", 
                               ["None", "Iris", "Wine", "Breast Cancer", 
                                "Synthetic Classification", "California Housing", "Synthetic Blobs"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("✅ Data uploaded successfully!")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head())
            st.session_state["df"] = df
        except Exception as e:
            st.error(f"Error loading file: {e}")

    elif demo_choice != "None":
        if demo_choice in ["California Housing"]:
            df = load_demo_dataset(demo_choice, task="regression")
        elif demo_choice in ["Synthetic Blobs"]:
            df = load_demo_dataset(demo_choice, task="clustering")
        else:
            df = load_demo_dataset(demo_choice, task="classification")
        st.write(f"✅ Loaded {demo_choice} dataset")
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head())
        st.session_state["df"] = df

# -------------------- EDA -------------------- #
elif options == "Exploratory Data Analysis":
    st.header("🔍 Exploratory Data Analysis (EDA)")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload or load a dataset first.")
    else:
        df = st.session_state["df"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Info")
            st.write(f"**Rows:** {df.shape[0]}")
            st.write(f"**Columns:** {df.shape[1]}")
            st.write("**Data Types:**")
            st.write(df.dtypes)
        
        with col2:
            st.subheader("Missing Values")
            missing = df.isnull().sum()
            if missing.sum() > 0:
                st.write(missing[missing > 0])
            else:
                st.success("No missing values!")
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("No numeric columns for correlation analysis")

# -------------------- REGRESSION -------------------- #
elif options == "Regression":
    st.header("📈 Regression")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload/load dataset first.")
    else:
        df = st.session_state["df"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns for regression")
        else:
            target = st.selectbox("Select target variable", numeric_cols)

            if st.button("Run Regression"):
                try:
                    X = df[numeric_cols].drop(columns=[target])
                    y = df[target]
                    
                    # Handle missing values
                    X = X.fillna(X.mean())
                    y = y.fillna(y.mean())
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    st.success("✅ Model trained successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Mean Squared Error", f"{mean_squared_error(y_test, preds):.4f}")
                    with col2:
                        st.metric("R² Score", f"{r2_score(y_test, preds):.4f}")
                    
                    # Predictions vs Actual plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_test, preds, alpha=0.6)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    ax.set_title("Predictions vs Actual Values")
                    st.pyplot(fig)
                    plt.close()

                    st.session_state["last_model"] = model
                    st.session_state["feature_names"] = X.columns.tolist()
                    
                except Exception as e:
                    st.error(f"Error during regression: {e}")

# -------------------- CLASSIFICATION -------------------- #
elif options == "Classification":
    st.header("🔢 Classification")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload/load dataset first.")
    else:
        df = st.session_state["df"]
        target = st.selectbox("Select target variable", df.columns)

        clf_choice = st.selectbox("Select classifier", 
                                  ["Logistic Regression", "Decision Tree", "Random Forest", 
                                   "KNN", "SVM", "Naive Bayes", "Gradient Boosting", "Voting Classifier"])

        if st.button("Run Classification"):
            try:
                X = df.drop(columns=[target])
                y = df[target]
                
                # Keep only numeric columns
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                X = X[numeric_cols]
                
                # Handle missing values
                X = X.fillna(X.mean())
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if clf_choice == "Logistic Regression":
                    model = LogisticRegression(max_iter=500)
                elif clf_choice == "Decision Tree":
                    model = DecisionTreeClassifier()
                elif clf_choice == "Random Forest":
                    model = RandomForestClassifier()
                elif clf_choice == "KNN":
                    model = KNeighborsClassifier()
                elif clf_choice == "SVM":
                    model = SVC()
                elif clf_choice == "Naive Bayes":
                    model = GaussianNB()
                elif clf_choice == "Gradient Boosting":
                    model = GradientBoostingClassifier()
                elif clf_choice == "Voting Classifier":
                    model = VotingClassifier(estimators=[
                        ('lr', LogisticRegression(max_iter=500)),
                        ('rf', RandomForestClassifier()),
                        ('gnb', GaussianNB())
                    ], voting='hard')

                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                accuracy = (preds == y_test).mean()

                st.success("✅ Model trained successfully!")
                st.metric("Accuracy", f"{accuracy:.2%}")
                
                st.subheader("Classification Report")
                st.text(classification_report(y_test, preds))

                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(y_test, preds)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                plt.close()

                st.session_state["last_model"] = model
                st.session_state["feature_names"] = X.columns.tolist()
                
            except Exception as e:
                st.error(f"Error during classification: {e}")

# -------------------- CLUSTERING -------------------- #
elif options == "Clustering":
    st.header("📉 Clustering")
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload/load dataset first.")
    else:
        df = st.session_state["df"]
        n_clusters = st.slider("Select number of clusters", 2, 10, 3)

        if st.button("Run KMeans Clustering"):
            try:
                X = df.select_dtypes(include=[np.number])
                
                if X.empty:
                    st.error("No numeric columns found for clustering")
                else:
                    # Handle missing values
                    X = X.fillna(X.mean())
                    
                    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                    labels = model.fit_predict(X)

                    st.success("✅ Clustering Complete")
                    df_clustered = df.copy()
                    df_clustered["Cluster"] = labels
                    st.dataframe(df_clustered.head(10))
                    
                    # Cluster distribution
                    st.subheader("Cluster Distribution")
                    cluster_counts = pd.Series(labels).value_counts().sort_index()
                    st.bar_chart(cluster_counts)
                    
                    # If 2D visualization is possible
                    if X.shape[1] >= 2:
                        st.subheader("Cluster Visualization (First 2 Features)")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', alpha=0.6)
                        ax.set_xlabel(X.columns[0])
                        ax.set_ylabel(X.columns[1])
                        plt.colorbar(scatter, ax=ax, label='Cluster')
                        st.pyplot(fig)
                        plt.close()

                    st.session_state["last_model"] = model
                    st.session_state["feature_names"] = X.columns.tolist()
                    
            except Exception as e:
                st.error(f"Error during clustering: {e}")

# -------------------- MODEL DEPLOYMENT -------------------- #
elif options == "Model Deployment":
    st.header("🚀 Model Deployment")

    if "last_model" in st.session_state:
        st.success("✅ Using last trained model from this session")
        model = st.session_state["last_model"]
        
        if "feature_names" in st.session_state:
            st.info(f"**Expected features:** {', '.join(st.session_state['feature_names'])}")
    else:
        st.warning("⚠️ No trained model available in session, please train one first.")
        model = None

    if model:
        pred_choice = st.radio("Choose input method:", ["Upload CSV", "Manual Input"])

        if pred_choice == "Upload CSV":
            uploaded_data = st.file_uploader("Upload CSV for prediction", type=["csv"])
            if uploaded_data is not None:
                try:
                    new_df = pd.read_csv(uploaded_data)
                    st.write("Input Data Preview:")
                    st.dataframe(new_df.head())

                    if st.button("Run Prediction"):
                        # Keep only numeric columns if needed
                        numeric_cols = new_df.select_dtypes(include=[np.number]).columns
                        X_pred = new_df[numeric_cols].fillna(new_df[numeric_cols].mean())
                        
                        predictions = model.predict(X_pred)
                        result_df = new_df.copy()
                        result_df["Predictions"] = predictions
                        
                        st.success("✅ Predictions Complete")
                        st.dataframe(result_df.head())

                        csv = result_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

        elif pred_choice == "Manual Input":
            st.info("Enter feature values manually for prediction")
            
            if "feature_names" in st.session_state:
                features = st.session_state["feature_names"]
                input_data = {}
                
                cols = st.columns(2)
                for i, feat in enumerate(features):
                    with cols[i % 2]:
                        input_data[feat] = st.number_input(f"{feat}", value=0.0, key=feat)
                
                if st.button("Predict"):
                    try:
                        features_array = np.array([input_data[f] for f in features]).reshape(1, -1)
                        prediction = model.predict(features_array)
                        st.success(f"✅ Prediction: **{prediction[0]}**")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
            else:
                st.warning("Feature names not available. Please train a model first.")