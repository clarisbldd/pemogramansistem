import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    data = pd.read_csv('car_evaluation_with.csv')
    le = LabelEncoder()
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = le.fit_transform(data[col])
    return data

# Fungsi untuk preprocessing dan split data
@st.cache_data
def preprocess_data(df):
    y = df['unacc']
    x = df.drop(columns=['unacc'])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test

# Fungsi untuk grid search pada SVM
def optimize_svm(x_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)
    return grid.best_estimator_

# Fungsi utama aplikasi
def main():
    st.title("Car Evaluation Binary Classification with Optimization")
    st.sidebar.title("Optimization Settings")
    st.markdown("An enhanced model to predict car acceptability 🚗")

    # Memuat dan memproses data
    df = load_data()
    x_train, x_test, y_train, y_test = preprocess_data(df)

    # Sidebar untuk pengaturan model
    st.sidebar.subheader("Optimization Options")
    if st.sidebar.button("Run Optimization"):
        st.subheader("Support Vector Machine (SVM) with Optimization")
        best_model = optimize_svm(x_train, y_train)
        
        # Evaluasi model
        y_pred = best_model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')

        st.write(f"Best Model Parameters: {best_model}")
        st.write(f"Accuracy: {acc:.4f}")
        st.write(f"Precision: {prec:.4f}")
        st.write(f"Recall: {rec:.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay.from_estimator(best_model, x_test, y_test, ax=ax, display_labels=['unacc', 'acc', 'good', 'vgood'])
        st.pyplot(fig)

        # Cross-Validation
        st.subheader("Cross-Validation Results")
        scores = cross_val_score(best_model, x_train, y_train, cv=5, scoring='accuracy')
        st.write(f"Cross-Validated Accuracy: {scores.mean():.4f}")

    # Opsi untuk menampilkan data mentah
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Car Evaluation Data")
        st.write(df)

# Menjalankan aplikasi
if __name__ == '__main__':
    main()
