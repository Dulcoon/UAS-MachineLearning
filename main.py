import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Function to handle missing values
def handle_missing_values(df, features):
    # Separate numerical and categorical features
    num_features = df[features].select_dtypes(include=['int64', 'float64']).columns
    cat_features = df[features].select_dtypes(include=['object']).columns

    # Impute numerical features with mean
    if not num_features.empty:
        num_imputer = SimpleImputer(strategy='mean')
        df[num_features] = num_imputer.fit_transform(df[num_features])

    # Impute categorical features with mode
    if not cat_features.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_features] = cat_imputer.fit_transform(df[cat_features])
    
    return df

# Function to perform Label Encoding
def preprocess_data(df, features):
    df_encoded = pd.get_dummies(df[features], drop_first=True)
    return df_encoded

def train_and_evaluate(df, features, label, model_params, test_size):
    df = handle_missing_values(df, features)  # Handle missing values before preprocessing
    X = preprocess_data(df, features)
    
    # Encoding the target label to numeric values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if model_params['model'] == 'KNN':
        model = KNeighborsClassifier(**model_params['params'])
    elif model_params['model'] == 'DecisionTree':
        model = DecisionTreeClassifier(**model_params['params'])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score,
        "Precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
        "Recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
        "F1 Score": lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
        "ROC AUC": lambda y_true, y_pred: roc_auc_score(y_true, model.predict_proba(X_test), multi_class='ovr', average='weighted') if len(np.unique(y)) > 2 else roc_auc_score(y_true, model.predict_proba(X_test)[:, 1])
    }

    results = {}
    for metric_name, metric in metrics.items():
        try:
            if metric_name == "ROC AUC" and len(np.unique(y)) > 2:
                results[metric_name] = metric(y_test, model.predict_proba(X_test))
            else:
                results[metric_name] = metric(y_test, y_pred)
        except ValueError as e:
            st.warning(f"Could not calculate {metric_name}: {e}")

    return results, X_train, X_test, y_train, y_test

def generate_pdf(results_knn, results_dt, params_knn, params_dt, X_train, X_test, y_train, y_test):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf_path = tmp.name
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica", 12)
        
        # Title
        c.drawString(30, 750, "Comparison of KNN and Decision Tree Models")
        
        # Metrics
        c.drawString(30, 735, "Metrics")
        c.drawString(150, 735, "KNN")
        c.drawString(250, 735, "Decision Tree")

        y_position = 720
        for metric in results_knn.keys():
            c.drawString(30, y_position, metric)
            c.drawString(150, y_position, str(round(results_knn[metric], 4)))
            c.drawString(250, y_position, str(round(results_dt[metric], 4)))
            y_position -= 15
        
        # Parameters
        y_position -= 20
        c.drawString(30, y_position, "KNN Parameters:")
        y_position -= 15
        for param, value in params_knn['params'].items():
            c.drawString(30, y_position, f"{param}: {value}")
            y_position -= 15
        
        y_position -= 20
        c.drawString(30, y_position, "Decision Tree Parameters:")
        y_position -= 15
        for param, value in params_dt['params'].items():
            c.drawString(30, y_position, f"{param}: {value}")
            y_position -= 15
        
        # Data
        y_position -= 20
        c.drawString(30, y_position, "Data:")
        y_position -= 15
        c.drawString(30, y_position, f"Train size: {X_train.shape}")
        y_position -= 15
        c.drawString(30, y_position, f"Test size: {X_test.shape}")
        y_position -= 15
        c.drawString(30, y_position, f"Train labels size: {y_train.shape}")
        y_position -= 15
        c.drawString(30, y_position, f"Test labels size: {y_test.shape}")

        c.save()
        return pdf_path

def plot_comparison(results_knn, results_dt):
    metrics = list(results_knn.keys())
    knn_scores = list(results_knn.values())
    dt_scores = list(results_dt.values())

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, knn_scores, width, label='KNN')
    rects2 = ax.bar(x + width/2, dt_scores, width, label='Decision Tree')

    ax.set_ylabel('Scores')
    ax.set_title('Comparison of KNN and Decision Tree')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 4)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    st.pyplot(fig)

st.title('Perbandingan Performansi Model Machine Learning: KNN vs Decision Tree')

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview", df.head())

    missing_values_count = df.isnull().sum().sum()
    if missing_values_count > 0:
        st.markdown(f"<span style='color:red'>{missing_values_count} missing value ditemukkan❗</span><br><span>Program ini sudah dilengkapi dengan penanganan Missing Value, silahkan melanjutkan proses!</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:#00d26a'>Tidak ada missing value dalam dataset✅</span>", unsafe_allow_html=True)

    all_columns = df.columns.tolist()

    st.markdown("---")

    selected_features = st.multiselect('Select Features', all_columns)
    selected_label = st.selectbox('Select Label', all_columns)

    with st.expander("Data Split"):
        train_size = st.number_input('Train Size', min_value=0.1, max_value=0.9, value=0.8, step=0.1)
        test_size = st.number_input('Test Size', min_value=0.1, max_value=0.9, value=0.2, step=0.1)

        # Ensure the sum of train_size and test_size is 1.0
        if train_size + test_size != 1.0:
            st.error("The sum of Train Size and Test Size must be 1.0")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("KNN Parameters"):
            n_neighbors = st.number_input('Number of Neighbors (n_neighbors)', min_value=1, value=5)
            weights = st.selectbox('Weights', ['uniform', 'distance'])
            algorithm = st.selectbox('Algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
            params_knn = {'model': 'KNN', 'params': {'n_neighbors': n_neighbors, 'weights': weights, 'algorithm': algorithm}}

    with col2:
        with st.expander("Decision Tree Parameters"):
            criterion = st.selectbox('Criterion', ['gini', 'entropy'])
            max_depth = st.number_input('Maximum Depth (max_depth)', min_value=1, value=5)
            min_samples_leaf = st.number_input('Minimum Samples per Leaf (min_samples_leaf)', min_value=1, value=1)
            min_samples_split = st.number_input('Minimum Samples per Split (min_samples_split)', min_value=2, value=2)
            params_dt = {'model': 'DecisionTree', 'params': {'criterion': criterion, 'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split}}

    if st.button('Compare Models'):
        if selected_features and selected_label and train_size + test_size == 1.0:
            results_knn, X_train, X_test, y_train, y_test = train_and_evaluate(df, selected_features, selected_label, params_knn, test_size)
            results_dt, _, _, _, _ = train_and_evaluate(df, selected_features, selected_label, params_dt, test_size)

            st.write('KNN Results:', results_knn)
            st.write('Decision Tree Results:', results_dt)

            comparison_df = pd.DataFrame({
                'Metric': list(results_knn.keys()),
                'KNN': list(results_knn.values()),
                'Decision Tree': list(results_dt.values())
            })

            st.write(comparison_df)

            plot_comparison(results_knn, results_dt)

            pdf_path = generate_pdf(results_knn, results_dt, params_knn, params_dt, X_train, X_test, y_train, y_test)
            with open(pdf_path, "rb") as file:
                st.download_button(
                    label="Download comparison as PDF",
                    data=file,
                    file_name="model_comparison.pdf",
                    mime="application/octet-stream"
                )
        else:
            st.error("Please select features, label, and ensure the sum of Train Size and Test Size is 1.0.")

st.write('---')
st.markdown('<div style="width: auto; margin: 0 auto; text-align: center;">'
            '<span style="font-weight: bolder";>--Kelompok 2--</span><br>Michael Valensio (5220411200)<br>Dionisius Lexy (5220411194)<br>Theodorus Richard Dakhi (5220411195)<br>Melky Sinun Usen (5220411190)<br>Claodio Nova (5220411185)'
            '</div>',
            unsafe_allow_html=True)
