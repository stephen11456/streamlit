import json
import os
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import modul Machine Learning dari scikit-learn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# Clustering Models & Metrics
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Koneksi MongoDB bersifat opsional saat dijalankan lokal
try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except ImportError:
    MongoClient = None
    PyMongoError = Exception

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
st.markdown('<div class="sub-title">Eksplorasi Regresi, Klasifikasi, dan Clustering secara visual, plus dataset & komentar tersimpan di MongoDB.</div>', unsafe_allow_html=True)

# ==========================================
# 2. KONFIGURASI MONGODB
# ==========================================
DB_NAME = "streamlit_ml"
MONGO_HINT = """
MongoDB belum terhubung. Pasang `pymongo`, lalu isi salah satu:

```toml
# .streamlit/secrets.toml
[mongodb]
uri = "mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true&w=majority"
```

atau gunakan environment variable `MONGODB_URI`.
"""

MODEL_OPTIONS = {
    "Regresi": [
        "Linear Regression", "Ridge Regression", "Lasso Regression",
        "Decision Tree Regressor", "Random Forest Regressor"
    ],
    "Klasifikasi": [
        "Logistic Regression", "K-Nearest Neighbors (KNN)",
        "Decision Tree Classifier", "Random Forest Classifier",
        "Support Vector Machine (SVM)"
    ],
    "Clustering": ["K-Means", "DBSCAN", "Agglomerative Clustering"],
}

DATASET_OPTIONS = {
    "Regresi": ["California Housing", "Diabetes", "Linnerud"],
    "Klasifikasi": ["Iris", "Wine", "Breast Cancer"],
    "Clustering": ["Iris", "Wine", "Breast Cancer"],
}


def uri_mongo():
    """Ambil URI MongoDB dari Streamlit Secrets atau environment variable."""
    try:
        bagian = st.secrets.get("mongodb", {})
        uri = bagian.get("uri") if bagian else None
    except Exception:
        uri = None
    return uri or os.environ.get("MONGODB_URI")


@st.cache_resource
def koleksi_mongo(nama_koleksi):
    """Hubungkan ke koleksi MongoDB. Mengembalikan None bila belum terkonfigurasi."""
    if MongoClient is None:
        return None
    uri = uri_mongo()
    if not uri:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client[DB_NAME][nama_koleksi]
    except PyMongoError:
        return None


# ==========================================
# 3. CACHING SCRIPT (LOAD DATASET)
# ==========================================
@st.cache_data
def load_dataset(task_type, dataset_name):
    """Muat dataset bawaan sklearn (Iris/Wine/Breast Cancer berasal dari UCI)."""
    if task_type == "Regresi":
        if dataset_name == "California Housing":
            data = datasets.fetch_california_housing(as_frame=True)
        elif dataset_name == "Diabetes":
            data = datasets.load_diabetes(as_frame=True)
        else:
            data = datasets.load_linnerud(as_frame=True)
            return data.data, data.target.iloc[:, 0], data.DESCR

    elif task_type == "Klasifikasi":
        if dataset_name == "Iris":
            data = datasets.load_iris(as_frame=True)
        elif dataset_name == "Wine":
            data = datasets.load_wine(as_frame=True)
        else:
            data = datasets.load_breast_cancer(as_frame=True)

    else:  # Clustering memakai fitur saja, tanpa label target
        if dataset_name == "Iris":
            data = datasets.load_iris(as_frame=True)
        elif dataset_name == "Wine":
            data = datasets.load_wine(as_frame=True)
        else:
            data = datasets.load_breast_cancer(as_frame=True)
        return data.data, None, data.DESCR

    return data.data, data.target, data.DESCR


# ==========================================
# 4. MODEL, PELATIHAN, DAN SESSION STATE
# ==========================================
def buat_model(task_type, model_name, n_clusters=3):
    """Buat estimator scikit-learn sesuai pilihan pengguna."""
    if task_type == "Regresi":
        if model_name == "Linear Regression":
            return LinearRegression()
        if model_name == "Ridge Regression":
            return Ridge()
        if model_name == "Lasso Regression":
            return Lasso()
        if model_name == "Decision Tree Regressor":
            return DecisionTreeRegressor(random_state=42)
        return RandomForestRegressor(random_state=42)

    if task_type == "Klasifikasi":
        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=1000)
        if model_name == "K-Nearest Neighbors (KNN)":
            return KNeighborsClassifier()
        if model_name == "Decision Tree Classifier":
            return DecisionTreeClassifier(random_state=42)
        if model_name == "Random Forest Classifier":
            return RandomForestClassifier(random_state=42)
        return SVC()

    if model_name == "K-Means":
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    if model_name == "DBSCAN":
        return DBSCAN(eps=0.5, min_samples=5)
    return AgglomerativeClustering(n_clusters=n_clusters)


def simpan_hasil(task_type, model_name, dataset_name, metrics, **ekstra):
    """Simpan hasil training ke session_state agar tetap ada saat rerun."""
    st.session_state.hasil = {
        "task": task_type,
        "model": model_name,
        "dataset": dataset_name,
        "metrics": metrics,
        **ekstra,
    }
    st.session_state.trained = True


def reset_hasil_jika_berubah(pilihan):
    """Hapus hasil lama ketika pengguna mengganti task/dataset/model/test size."""
    if st.session_state.get("pilihan_terakhir") != pilihan:
        st.session_state.trained = False
        st.session_state.hasil = None
        st.session_state.pilihan_terakhir = pilihan


def latih_dan_simpan(task_type, model_name, X, y, test_size, dataset_name, n_clusters=3):
    """Latih model, hitung metrik, lalu simpan hasil ke session_state."""
    if task_type == "Clustering":
        scaler = StandardScaler()
        X_proses = scaler.fit_transform(X)
        model = buat_model(task_type, model_name, n_clusters)
        labels = model.fit_predict(X_proses)

        jumlah_cluster = len(set(labels)) - (1 if -1 in labels else 0)
        metrik = {"Jumlah Cluster": jumlah_cluster}
        tanpa_noise = labels != -1

        if 1 < jumlah_cluster < int(tanpa_noise.sum()):
            metrik.update({
                "Silhouette": silhouette_score(X_proses[tanpa_noise], labels[tanpa_noise]),
                "Calinski-Harabasz": calinski_harabasz_score(X_proses[tanpa_noise], labels[tanpa_noise]),
                "Davies-Bouldin": davies_bouldin_score(X_proses[tanpa_noise], labels[tanpa_noise]),
            })

        pca = PCA(n_components=2, random_state=42)
        X_plot = pca.fit_transform(X_proses)
        simpan_hasil(
            task_type, model_name, dataset_name, metrik,
            labels=labels, X_plot=X_plot, noise=int((labels == -1).sum())
        )
        if jumlah_cluster <= 1:
            st.warning("⚠️ DBSCAN menghasilkan ≤ 1 cluster. Coba ubah `eps` atau `min_samples` di kode.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    model = buat_model(task_type, model_name, n_clusters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task_type == "Regresi":
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        metrik = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R²": r2_score(y_test, y_pred),
        }
        simpan_hasil(task_type, model_name, dataset_name, metrik, y_test=y_test, y_pred=y_pred)
        return

    metrik = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    simpan_hasil(
        task_type, model_name, dataset_name, metrik,
        report=classification_report(y_test, y_pred),
        cm=confusion_matrix(y_test, y_pred),
        y_test=y_test, y_pred=y_pred,
    )


# ==========================================
# 5. SIDEBAR / NAVIGASI SELEKSI MODEL & DATASET
# ==========================================
st.sidebar.header("⚙️ Pengaturan Pembelajaran")

task_type = st.sidebar.radio(
    "📌 Pilih Jenis Tugas (Task):",
    list(MODEL_OPTIONS.keys())
)
dataset_name = st.sidebar.selectbox(
    f"📊 Pilih Dataset {task_type}:",
    DATASET_OPTIONS[task_type]
)
model_name = st.sidebar.selectbox(
    f"🧠 Pilih Algoritma {task_type}:",
    MODEL_OPTIONS[task_type]
)

n_clusters = 3
if task_type == "Clustering":
    n_clusters = st.sidebar.slider("🔢 Jumlah Cluster (K-Means / Agglomerative):", 2, 8, 3)

test_size = st.sidebar.slider(
    "✂️ Ukuran Data Uji (Test Size):",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05,
    help="Persentase dataset yang digunakan untuk evaluasi. Tidak dipakai pada Clustering."
)

train_button = st.sidebar.button("🚀 Latih Model Sekarang", type="primary", use_container_width=True)
st.sidebar.caption("📝 Daftarkan URL app: mdsfsmuksw.my.id/materi-belajar/streamlit")

reset_hasil_jika_berubah((task_type, dataset_name, model_name, test_size, n_clusters))

# ==========================================
# 6. MEMUAT DATASET DAN INFO SINGKAT
# ==========================================
X, y, description = load_dataset(task_type, dataset_name)

if train_button:
    latih_dan_simpan(task_type, model_name, X, y, test_size, dataset_name, n_clusters)

with st.expander("ℹ️ Informasi Ringkas Dataset Yang Dipilih"):
    st.write(f"**Nama Dataset:** {dataset_name} | **Jumlah Baris:** {X.shape[0]} | **Jumlah Fitur:** {X.shape[1]}")
    st.caption("Sumber: scikit-learn toy datasets (Iris/Wine/Breast Cancer berasal dari UCI-repo).")
    st.dataframe(X.head(3), use_container_width=True)

# ==========================================
# 7. TAMPILAN TAB HASIL & METRIK
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Tampilan Visual",
    "📊 Metrik Evaluasi & Laporan",
    "🗄️ Dataset MongoDB & Retraining",
    "💬 Komentar Publik",
])

hasil = st.session_state.get("hasil")

# TAB 1: VISUALISASI
with tab1:
    if hasil:
        st.subheader(f"🖼️ Hasil Visualisasi — {hasil['task']} ({hasil['model']})")
        fig, ax = plt.subplots(figsize=(8, 5))

        if hasil["task"] == "Regresi":
            sns.scatterplot(
                x=hasil["y_test"], y=hasil["y_pred"],
                alpha=0.7, color="#1E88E5", ax=ax
            )
            min_val = min(min(hasil["y_test"]), min(hasil["y_pred"]))
            max_val = max(max(hasil["y_test"]), max(hasil["y_pred"]))
            ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Garis Prediksi Ideal")
            ax.set_xlabel("Nilai Sebenarnya (Actual)")
            ax.set_ylabel("Nilai Hasil Prediksi (Predicted)")
            ax.set_title("Grafik Actual vs Predicted")
            ax.legend()

        elif hasil["task"] == "Klasifikasi":
            sns.heatmap(hasil["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel("Kelas Hasil Prediksi")
            ax.set_ylabel("Kelas Sebenarnya")
            ax.set_title("Matriks Kebingungan (Confusion Matrix)")

        else:
            plot_df = pd.DataFrame({
                "PC1": hasil["X_plot"][:, 0],
                "PC2": hasil["X_plot"][:, 1],
                "Cluster": hasil["labels"].astype(str),
            })
            sns.scatterplot(
                data=plot_df, x="PC1", y="PC2", hue="Cluster",
                palette="tab10", s=60, alpha=0.8, ax=ax
            )
            ax.set_title(f"Visualisasi Cluster — {hasil['model']} (PCA 2D)")
            st.caption(f"🔍 Titik noise (label -1): {hasil.get('noise', 0)}")

        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("👈 Silakan tekan tombol **'🚀 Latih Model Sekarang'** pada Sidebar untuk memulai pelatihan dan melihat visualisasi!")

# TAB 2: METRIK EVALUASI
with tab2:
    if hasil:
        st.subheader("📊 Metrik Performa Model")
        metrik = hasil["metrics"]
        cols = st.columns(len(metrik))
        for idx, (nama, nilai) in enumerate(metrik.items()):
            if isinstance(nilai, (int, np.integer)):
                cols[idx].metric(label=nama, value=int(nilai))
            else:
                cols[idx].metric(label=nama, value=f"{nilai:.4f}")

        st.markdown("---")
        if hasil["task"] == "Klasifikasi":
            st.subheader("📋 Laporan Klasifikasi Detil (Classification Report)")
            st.code(hasil["report"], language="text")
        elif hasil["task"] == "Clustering":
            st.caption("ℹ️ Metrik clustering dihitung tanpa label sebenarnya (unsupervised).")
    else:
        st.info("👈 Silakan tekan tombol **'🚀 Latih Model Sekarang'** pada Sidebar untuk melihat metrik evaluasi!")

# TAB 3: DATASET MONGODB & RETRAINING
with tab3:
    st.subheader("🗄️ Tambah Dataset ke MongoDB & Latih Ulang")

    koleksi_dataset = koleksi_mongo("custom_datasets")
    if koleksi_dataset is None:
        st.warning(MONGO_HINT)

    st.markdown("#### 📂 Unggah Dataset Baru")
    uploaded = st.file_uploader("Pilih file CSV", type=["csv"], key="upload_csv")
    task_baru = st.selectbox("🎯 Task untuk dataset ini", list(MODEL_OPTIONS), key="task_baru")
    nama_baru = st.text_input("🏷️ Nama dataset", value="dataset_saya", key="nama_baru")
    target_baru = st.text_input(
        "🎯 Kolom target (kosongkan untuk Clustering)",
        value="", key="target_baru"
    )

    if uploaded is not None:
        try:
            df_baru = pd.read_csv(uploaded)
            st.dataframe(df_baru.head(), use_container_width=True)
        except Exception as error:
            st.error(f"Gagal membaca CSV: {error}")
        else:
            if st.button("💾 Simpan Dataset ke MongoDB", type="primary", key="simpan_dataset"):
                if task_baru != "Clustering" and target_baru not in df_baru.columns:
                    st.error(f"Kolom target '{target_baru}' tidak ditemukan. Kolom tersedia: {list(df_baru.columns)}")
                elif koleksi_dataset is None:
                    st.warning(MONGO_HINT)
                else:
                    dokumen = {
                        "nama": nama_baru,
                        "task": task_baru,
                        "target": target_baru or None,
                        "columns": list(df_baru.columns),
                        "rows": len(df_baru),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "data": json.loads(df_baru.to_json(orient="records")),
                    }
                    try:
                        koleksi_dataset.insert_one(dokumen)
                        st.success(f"✅ Dataset '{nama_baru}' berhasil disimpan ke MongoDB.")
                    except PyMongoError as error:
                        st.error(f"Gagal menyimpan ke MongoDB: {error}")

    st.markdown("---")
    st.markdown("#### 🔁 Retrain dari Dataset Tersimpan")

    if koleksi_dataset is None:
        st.info("Koneksi MongoDB belum tersedia, sehingga daftar dataset tidak dapat dimuat.")
    else:
        try:
            daftar_dataset = list(
                koleksi_dataset.find({}, {"_id": 0, "nama": 1, "task": 1, "target": 1, "rows": 1, "created_at": 1})
            )
        except PyMongoError as error:
            daftar_dataset = []
            st.error(f"Gagal memuat daftar dataset: {error}")

        if not daftar_dataset:
            st.info("Belum ada dataset tersimpan. Unggah CSV di atas terlebih dahulu.")
        else:
            st.dataframe(pd.DataFrame(daftar_dataset), use_container_width=True)
            nama_tersimpan = st.selectbox(
                "Pilih dataset tersimpan",
                [item.get("nama", "") for item in daftar_dataset],
                key="pilih_dataset"
            )
            dokumen = koleksi_dataset.find_one({"nama": nama_tersimpan}, {"_id": 0})
            task_tersimpan = dokumen.get("task", "Klasifikasi")
            if task_tersimpan not in MODEL_OPTIONS:
                task_tersimpan = "Klasifikasi"
            model_tersimpan = st.selectbox(
                "Model untuk retrain",
                MODEL_OPTIONS[task_tersimpan],
                key="model_retrain"
            )

            if st.button("🚀 Latih Ulang Model dari MongoDB", type="primary", key="retrain"):
                df_mongo = pd.DataFrame(dokumen.get("data", []))
                if df_mongo.empty:
                    st.error("Dataset tersimpan kosong.")
                else:
                    if task_tersimpan == "Clustering":
                        X_mongo = df_mongo.select_dtypes(include=[np.number])
                        y_mongo = None
                    else:
                        target = dokumen.get("target")
                        if target not in df_mongo.columns:
                            st.error(f"Kolom target '{target}' tidak ditemukan pada dataset tersimpan.")
                            st.stop()
                        X_mongo = pd.get_dummies(df_mongo.drop(columns=[target]), drop_first=True)
                        y_mongo = df_mongo[target]

                    if X_mongo.shape[1] == 0:
                        st.error("Tidak ada fitur numerik yang bisa digunakan untuk training.")
                    else:
                        latih_dan_simpan(
                            task_tersimpan, model_tersimpan, X_mongo, y_mongo,
                            test_size, f"MongoDB: {nama_tersimpan}"
                        )
                        st.success("✅ Retraining selesai! Lihat hasilnya di Tab 1 dan Tab 2.")

# TAB 4: KOMENTAR PUBLIK
with tab4:
    st.subheader("💬 Komentar Publik (Tersimpan di MongoDB)")

    koleksi_komentar = koleksi_mongo("comments")
    if koleksi_komentar is None:
        st.warning(MONGO_HINT)

    with st.form("form_komentar", clear_on_submit=True):
        nama_komentar = st.text_input("👤 Nama kamu", value="Publik")
        isi_komentar = st.text_area("✍️ Tulis komentar", height=90)
        kirim_komentar = st.form_submit_button("📤 Kirim Komentar")

    if kirim_komentar:
        if not isi_komentar.strip():
            st.warning("Komentar masih kosong.")
        else:
            dokumen_komentar = {
                "nama": nama_komentar.strip() or "Publik",
                "isi": isi_komentar.strip(),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            if koleksi_komentar is None:
                st.session_state.setdefault("komentar_lokal", []).append(dokumen_komentar)
                st.warning("⚠️ Komentar tersimpan sementara di session. MongoDB belum terhubung.")
            else:
                try:
                    koleksi_komentar.insert_one(dokumen_komentar)
                    st.success("✅ Komentar berhasil disimpan ke MongoDB.")
                except PyMongoError as error:
                    st.error(f"Gagal menyimpan komentar: {error}")

    st.markdown("---")
    st.markdown("#### 📜 Daftar Komentar")

    try:
        if koleksi_komentar is not None:
            daftar_komentar = list(
                koleksi_komentar.find({}, {"_id": 0}).sort("created_at", -1).limit(20)
            )
        else:
            daftar_komentar = list(reversed(st.session_state.get("komentar_lokal", [])))
    except PyMongoError as error:
        daftar_komentar = []
        st.error(f"Gagal memuat komentar: {error}")

    if not daftar_komentar:
        st.info("Belum ada komentar. Jadilah yang pertama berkomentar!")
    else:
        for komentar in daftar_komentar:
            st.markdown(
                f"**{komentar.get('nama', 'Publik')}** · _{komentar.get('created_at', '')}_\n\n"
                f"{komentar.get('isi', '')}"
            )
            st.divider()
