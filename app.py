import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import modul Machine Learning dari scikit-learn
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Regression Models & Metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Classification Models & Metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ==========================================
# 1. KONFIGURASI HALAMAN / PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Aplikasi Pembelajaran Mesin Interaktif",
    page_icon="🤖",
    layout="wide"
)

# Kostumisasi tampilan dengan CSS sederhana
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 Dasbor Machine Learning Scikit-Learn</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Eksplorasi dan bandingkan performa berbagai algoritma Regresi & Klasifikasi secara visual.</div>', unsafe_allow_html=True)

# ==========================================
# 2. CACHING SCRIPT (LOAD DATASET)
# ==========================================
@st.cache_data
def load_dataset(task_type, dataset_name):
    """
    Fungsi untuk memuat dataset bawaan sklearn berdasarkan pilihan task dan nama dataset.
    Menggunakan @st.cache_data agar performa cepat saat berpindah-pindah tab.
    """
    if task_type == "Regresi":
        if dataset_name == "California Housing":
            data = datasets.fetch_california_housing(as_frame=True)
        elif dataset_name == "Diabetes":
            data = datasets.load_diabetes(as_frame=True)
        elif dataset_name == "Linnerrud":
            data = datasets.load_linnerud(as_frame=True)
            X = data.data
            y = data.target.iloc[:, 0]
            return X, y, data.DESCR
    else:  # Klasifikasi
        if dataset_name == "Iris":
            data = datasets.load_iris(as_frame=True)
        elif dataset_name == "Wine":
            data = datasets.load_wine(as_frame=True)
        elif dataset_name == "Breast Cancer":
            data = datasets.load_breast_cancer(as_frame=True)
            
    X = data.data
    y = data.target
    return X, y, data.DESCR

# ==========================================
# 3. SIDEBAR / NAVIGASI SELEKSI MODEL & DATASET
# ==========================================
st.sidebar.header("⚙️ Pengaturan Pembelajaran")

# Pilih Tipe Tugas (Regresi atau Klasifikasi)
task_type = st.sidebar.radio(
    "📌 Pilih Jenis Tugas (Task):",
    ["Regresi", "Klasifikasi"]
)

# Pilihan Dataset berdasarkan Tugas
if task_type == "Regresi":
    dataset_name = st.sidebar.selectbox(
        "📊 Pilih Dataset Regresi:",
        ["California Housing", "Diabetes", "Linnerrud"]
    )
    model_name = st.sidebar.selectbox(
        "🧠 Pilih Algoritma Regresi:",
        ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree Regressor", "Random Forest Regressor"]
    )
else:
    dataset_name = st.sidebar.selectbox(
        "📊 Pilih Dataset Klasifikasi:",
        ["Iris", "Wine", "Breast Cancer"]
    )
    model_name = st.sidebar.selectbox(
        "🧠 Pilih Algoritma Klasifikasi:",
        ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Decision Tree Classifier", "Random Forest Classifier", "Support Vector Machine (SVM)"]
    )

# Slider Nilai Ukuran Data Uji (Test Size)
test_size = st.sidebar.slider(
    "✂️ Ukuran Data Uji (Test Size):",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05,
    help="Persentase dataset yang digunakan untuk evaluasi model."
)

# Tombol Latih Model
train_button = st.sidebar.button("🚀 Latih Model Sekarang", type="primary", use_container_width=True)

# ==========================================
# 4. MEMUAT DATASET DAN PREPARASI SESSION STATE
# ==========================================
X, y, description = load_dataset(task_type, dataset_name)

# Inisialisasi Session State jika belum ada
if "trained" not in st.session_state:
    st.session_state.trained = False

# ==========================================
# 5. LOGIKA PELATIHAN MODEL (TRAINING LOGIC)
# ==========================================
if train_button:
    # Split data menjadi Data Latih dan Data Uji
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Inisialisasi Model berdasarkan Pilihan
    if task_type == "Regresi":
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Ridge Regression":
            model = Ridge()
        elif model_name == "Lasso Regression":
            model = Lasso()
        elif model_name == "Decision Tree Regressor":
            model = DecisionTreeRegressor(random_state=42)
        elif model_name == "Random Forest Regressor":
            model = RandomForestRegressor(random_state=42)
            
        # Fitting Model & Prediksi
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Hitung Metrik Evaluasi Regresi
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Simpan ke Session State
        st.session_state.metrics = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2
        }
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred

    else:  # Klasifikasi
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "K-Nearest Neighbors (KNN)":
            model = KNeighborsClassifier()
        elif model_name == "Decision Tree Classifier":
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == "Random Forest Classifier":
            model = RandomForestClassifier(random_state=42)
        elif model_name == "Support Vector Machine (SVM)":
            model = SVC()

        # Fitting Model & Prediksi
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Hitung Metrik Evaluasi Klasifikasi
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        report = classification_report(y_test, y_pred, output_dict=False)
        cm = confusion_matrix(y_test, y_pred)

        # Simpan ke Session State
        st.session_state.metrics = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        }
        st.session_state.report = report
        st.session_state.cm = cm
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred

    # Simpan status bahwa pelatihan telah selesai
    st.session_state.trained = True
    st.session_state.current_task = task_type
    st.session_state.current_model = model_name
    st.session_state.current_dataset = dataset_name

# Informasi singkat dataset
with st.expander("ℹ️ Informasi Ringkas Dataset Yang Dipilih"):
    st.write(f"**Nama Dataset:** {dataset_name} | **Jumlah Baris:** {X.shape[0]} | **Jumlah Fitur:** {X.shape[1]}")
    st.dataframe(X.head(3), use_container_width=True)

# ==========================================
# 6. TAMPILAN TAB HASIL & METRIK
# ==========================================
tab1, tab2 = st.tabs(["📈 Tampilan Visual", "📊 Metrik Evaluasi & Laporan"])

# TAB 1: VISUALISASI
with tab1:
    if st.session_state.get("trained", False):
        st.subheader(f"🖼️ Hasil Visualisasi ({st.session_state.current_model})")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if st.session_state.current_task == "Regresi":
            # Plot Actual vs Predicted
            sns.scatterplot(
                x=st.session_state.y_test, 
                y=st.session_state.y_pred, 
                alpha=0.7, 
                color="#1E88E5", 
                ax=ax
            )
            # Garis prediksi ideal
            min_val = min(min(st.session_state.y_test), min(st.session_state.y_pred))
            max_val = max(max(st.session_state.y_test), max(st.session_state.y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Garis Prediksi Ideal")
            
            ax.set_xlabel("Nilai Sebenarnya (Actual)")
            ax.set_ylabel("Nilai Hasil Prediksi (Predicted)")
            ax.set_title("Grafik Actual vs Predicted")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            
        else:
            # Heatmap Confusion Matrix
            sns.heatmap(
                st.session_state.cm, 
                annot=True, 
                fmt="d", 
                cmap="Blues", 
                cbar=False, 
                ax=ax
            )
            ax.set_xlabel("Kelas Hasil Prediksi")
            ax.set_ylabel("Kelas Sebenarnya")
            ax.set_title("Matriks Kebingungan (Confusion Matrix)")
            
        st.pyplot(fig)
    else:
        st.info("👈 Silakan tekan tombol **'🚀 Latih Model Sekarang'** pada Sidebar untuk memulai pelatihan dan melihat visualisasi!")

# TAB 2: METRIK EVALUASI
with tab2:
    if st.session_state.get("trained", False):
        st.subheader("📊 Metrik Performa Model")
        
        metrics = st.session_state.metrics
        cols = st.columns(len(metrics))
        
        # Menampilkan metrik dalam bentuk Card Metric
        for idx, (m_name, m_val) in enumerate(metrics.items()):
            cols[idx].metric(label=m_name, value=f"{m_val:.4f}")
            
        st.markdown("---")
        
        # Tampilkan Classification Report jika tugasnya Klasifikasi
        if st.session_state.current_task == "Klasifikasi":
            st.subheader("📋 Laporan Klasifikasi Detil (Classification Report)")
            st.code(st.session_state.report, language="text")
            
    else:
        st.info("👈 Silakan tekan tombol **'🚀 Latih Model Sekarang'** pada Sidebar untuk melihat metrik evaluasi!")