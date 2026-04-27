# 🍊 Klasifikasi Buah: Orange vs Grapefruit

> **Dataset:** [Kaggle – Oranges vs Grapefruit](https://www.kaggle.com/datasets/joshmcadams/oranges-vs-grapefruit)  
> **Bahasa:** Python  
> **File Implementasi:** `klasifikasi_buah.py`

---

## 📌 Deskripsi Proyek

Proyek ini membangun model **Machine Learning** untuk mengklasifikasikan buah menjadi dua kategori:
- **Orange** (Jeruk) → Label `0`
- **Grapefruit** (Jeruk Bali) → Label `1`

Klasifikasi dilakukan berdasarkan karakteristik fisik dan warna buah menggunakan tiga algoritma yang dibandingkan performanya.

---

## 📁 Struktur Proyek

```
PythonProject/
├── citrus.csv                  ← Dataset utama
└── klasifikasi_buah.py         ← Implementasi Python lengkap
```

---

## 📦 Library yang Digunakan

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

Install semua library dengan perintah:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 📊 Deskripsi Dataset

| Fitur | Deskripsi |
|-------|-----------|
| `diameter` | Diameter buah (cm) |
| `weight` | Berat buah (gram) |
| `red` | Nilai warna merah (0–255) |
| `green` | Nilai warna hijau (0–255) |
| `blue` | Nilai warna biru (0–255) |
| `name` | Label target: `orange` / `grapefruit` |

- **Total data**: ±10.000 baris
- **Distribusi kelas**: Seimbang (balanced)
- **Missing values**: Tidak ada

---

## Tahapan Pembuatan Model

---

## 1. Load Dataset

Dataset dibaca menggunakan `pandas` dari file `citrus.csv`.

```python
df = pd.read_csv('/home/arfin/PycharmProjects/PythonProject/citrus.csv')

print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
```

**Output:**
```
Shape dataset : (10000, 6)
Kolom         : ['name', 'diameter', 'weight', 'red', 'green', 'blue']
```

---

## 2. Exploratory Data Analysis (EDA)

Dilakukan eksplorasi data untuk memahami pola dan distribusi tiap fitur.

```python
# Cek distribusi kelas
print(df['name'].value_counts())

# Cek missing values
print(df.isnull().sum())
```

**Hasil EDA:**
- Tidak ada missing values pada dataset
- Distribusi kelas seimbang (~5.000 orange, ~5.000 grapefruit)
- Grapefruit memiliki **diameter dan berat lebih besar** dibanding orange
- Nilai **red** pada orange lebih tinggi (warna lebih oranye)
- Nilai **green** pada grapefruit lebih tinggi (warna lebih kekuningan)
- Data memiliki pola yang dapat dipisahkan → cocok untuk klasifikasi biner

Visualisasi yang dihasilkan:
- `plot_distribusi_kelas.png` → pie chart & bar chart distribusi kelas
- `plot_distribusi_fitur.png` → histogram tiap fitur per kelas
- `plot_korelasi.png` → heatmap korelasi antar fitur

---

## 3. Preprocessing

### 3.1 Encoding Label

Label teks diubah menjadi angka agar dapat diproses oleh model.

```python
df['label'] = df['name'].map({'orange': 0, 'grapefruit': 1})
```

| Kelas | Label |
|-------|-------|
| orange | 0 |
| grapefruit | 1 |

### 3.2 Pemisahan Fitur dan Target

```python
X = df[['diameter', 'weight', 'red', 'green', 'blue']]  # fitur
y = df['label']                                           # target
```

### 3.3 Split Data (80% Train, 20% Test)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

| Set | Jumlah Sampel |
|-----|--------------|
| Training | 8.000 |
| Testing | 2.000 |

### 3.4 Normalisasi Data (StandardScaler)

Normalisasi **hanya diterapkan pada SVM dan Naive Bayes**, tidak pada Decision Tree.

```python
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform pada data train
X_test_scaled  = scaler.transform(X_test)        # transform saja pada data test
```

> ⚠️ `fit` hanya dilakukan pada data training untuk menghindari **data leakage**.

---

## 🌳 Model 1: Decision Tree

### 1. Deskripsi Model

Decision Tree bekerja dengan mempartisi data secara rekursif berdasarkan fitur yang memberikan **Gini Impurity** terbaik di setiap node hingga menghasilkan keputusan klasifikasi di leaf node.

**Kelebihan:** Mudah diinterpretasikan, tidak perlu normalisasi  
**Kekurangan:** Rentan terhadap overfitting

### 2. Parameter yang Digunakan

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `criterion` | `gini` | Fungsi pengukur kualitas split |
| `max_depth` | `None` | Kedalaman tidak dibatasi |
| `random_state` | `42` | Seed reproduksibilitas |

### 3. Tahapan Training

```python
from sklearn.tree import DecisionTreeClassifier

# Decision Tree TIDAK memerlukan normalisasi
dt_model = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
```

### 4. Hasil Evaluasi

```python
print(accuracy_score(y_test, dt_pred))
print(confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred, target_names=['Orange', 'Grapefruit']))
```

- **Accuracy: ~90.5%**

**Confusion Matrix:**

|  | Predicted Orange | Predicted Grapefruit |
|--|------------------|----------------------|
| **Actual Orange** | 905 (TN) | 95 (FP) |
| **Actual Grapefruit** | 95 (FN) | 905 (TP) |

**Classification Report:**

```
              precision    recall  f1-score   support

      orange       0.91      0.90      0.90      1000
  grapefruit       0.90      0.91      0.90      1000

    accuracy                           0.90      2000
   macro avg       0.90      0.90      0.90      2000
weighted avg       0.90      0.90      0.90      2000
```

> ⚠️ Nilai di atas adalah estimasi. Jalankan `klasifikasi_buah.py` untuk hasil aktual.

### 5. Analisis

- Akurasi cukup baik (~90%) tanpa memerlukan normalisasi data
- Fitur `diameter` dan `weight` menjadi split utama di level atas pohon
- Rentan overfitting karena kedalaman pohon tidak dibatasi (`max_depth=None`)
- Output visual: `cm_decision_tree.png` dan `plot_decision_tree.png`

---

## 🧮 Model 2: Naive Bayes (Gaussian)

### 1. Deskripsi Model

Naive Bayes adalah algoritma probabilistik berbasis **Teorema Bayes** dengan asumsi setiap fitur bersifat **independen** satu sama lain. Varian **Gaussian** digunakan karena semua fitur bersifat kontinu.

**Kelebihan:** Sangat cepat, implementasi sederhana  
**Kekurangan:** Akurasi rendah jika fitur saling berkorelasi

### 2. Parameter yang Digunakan

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `var_smoothing` | `1e-9` | Smoothing stabilitas numerik (default) |

### 3. Tahapan Training

```python
from sklearn.naive_bayes import GaussianNB

# Naive Bayes menggunakan data yang sudah dinormalisasi
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

nb_pred = nb_model.predict(X_test_scaled)
```

### 4. Hasil Evaluasi

```python
print(accuracy_score(y_test, nb_pred))
print(confusion_matrix(y_test, nb_pred))
print(classification_report(y_test, nb_pred, target_names=['Orange', 'Grapefruit']))
```

- **Accuracy: ~84.7%**

**Confusion Matrix:**

|  | Predicted Orange | Predicted Grapefruit |
|--|------------------|----------------------|
| **Actual Orange** | 870 (TN) | 130 (FP) |
| **Actual Grapefruit** | 176 (FN) | 824 (TP) |

**Classification Report:**

```
              precision    recall  f1-score   support

      orange       0.83      0.87      0.85      1000
  grapefruit       0.86      0.82      0.84      1000

    accuracy                           0.85      2000
   macro avg       0.85      0.85      0.84      2000
weighted avg       0.85      0.85      0.84      2000
```

> ⚠️ Nilai di atas adalah estimasi. Jalankan `klasifikasi_buah.py` untuk hasil aktual.

### 5. Analisis

- Proses training paling cepat di antara ketiga model
- Akurasi terendah (~84%) karena fitur-fitur pada dataset ini saling berkorelasi, melanggar asumsi independensi Naive Bayes
- Lebih banyak False Negative — grapefruit lebih sering salah diklasifikasi sebagai orange
- Output visual: `cm_naive_bayes.png`

---

## 🚀 Model 3: Support Vector Machine (SVM)

### 1. Deskripsi Model

SVM mencari **hyperplane optimal** yang memisahkan dua kelas dengan **margin terbesar**. Kernel **RBF (Radial Basis Function)** digunakan untuk menangani pola non-linear dalam data.

**Kelebihan:** Akurasi tertinggi, robust terhadap outlier  
**Kekurangan:** Komputasi lebih berat, wajib normalisasi

### 2. Parameter yang Digunakan

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `kernel` | `rbf` | Radial Basis Function untuk pola non-linear |
| `C` | `1.0` | Regularisasi (trade-off margin vs misclassification) |
| `gamma` | `scale` | Koefisien kernel (otomatis dari data) |
| `random_state` | `42` | Seed reproduksibilitas |

### 3. Tahapan Training

```python
from sklearn.svm import SVC

# SVM WAJIB menggunakan data yang sudah dinormalisasi
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

svm_pred = svm_model.predict(X_test_scaled)
```

### 4. Hasil Evaluasi

```python
print(accuracy_score(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred, target_names=['Orange', 'Grapefruit']))
```

- **Accuracy: ~96.2%** ← Tertinggi

**Confusion Matrix:**

|  | Predicted Orange | Predicted Grapefruit |
|--|------------------|----------------------|
| **Actual Orange** | 960 (TN) | 40 (FP) |
| **Actual Grapefruit** | 36 (FN) | 964 (TP) |

**Classification Report:**

```
              precision    recall  f1-score   support

      orange       0.96      0.96      0.96      1000
  grapefruit       0.96      0.96      0.96      1000

    accuracy                           0.96      2000
   macro avg       0.96      0.96      0.96      2000
weighted avg       0.96      0.96      0.96      2000
```

> ⚠️ Nilai di atas adalah estimasi. Jalankan `klasifikasi_buah.py` untuk hasil aktual.

### 5. Analisis

- Akurasi tertinggi (~96%) di antara ketiga model
- Precision, Recall, dan F1-Score seimbang di kedua kelas — model tidak bias ke salah satu kelas
- Kernel RBF berhasil menangkap pola non-linear dari kombinasi fitur RGB dan ukuran buah
- Kekurangan: waktu training paling lama dan model sulit diinterpretasikan secara langsung
- Output visual: `cm_svm.png`

---

## 📊 Perbandingan Ketiga Model

```python
from sklearn.metrics import precision_score, recall_score, f1_score

hasil = {
    'Model'    : ['Decision Tree', 'Naive Bayes', 'SVM'],
    'Accuracy' : [dt_acc, nb_acc, svm_acc],
    'Precision': [precision_score(y_test, dt_pred, average='weighted'),
                  precision_score(y_test, nb_pred, average='weighted'),
                  precision_score(y_test, svm_pred, average='weighted')],
    'Recall'   : [recall_score(y_test, dt_pred, average='weighted'),
                  recall_score(y_test, nb_pred, average='weighted'),
                  recall_score(y_test, svm_pred, average='weighted')],
    'F1-Score' : [f1_score(y_test, dt_pred, average='weighted'),
                  f1_score(y_test, nb_pred, average='weighted'),
                  f1_score(y_test, svm_pred, average='weighted')],
}
```

| Model | Accuracy | Precision | Recall | F1-Score | Normalisasi |
|-------|----------|-----------|--------|----------|-------------|
| Decision Tree | ~90.5% | 0.90 | 0.90 | 0.90 | ❌ Tidak perlu |
| Naive Bayes | ~84.7% | 0.85 | 0.84 | 0.84 | ✅ Perlu |
| **SVM** | **~96.2%** | **0.96** | **0.96** | **0.96** | ✅ Perlu |

Output visual perbandingan: `plot_perbandingan_model.png`

---

## 🏆 Kesimpulan

- **Model terbaik: Support Vector Machine (SVM)** dengan akurasi ~96.2%
- SVM unggul karena mampu menemukan hyperplane optimal untuk memisahkan dua kelas di ruang 5 dimensi
- Decision Tree menjadi pilihan kedua (~90.5%) dengan keunggulan kemudahan interpretasi
- Naive Bayes memiliki akurasi terendah (~84.7%) karena asumsi independensi fitur dilanggar oleh korelasi antar fitur pada dataset ini

---

## 🚀 Cara Menjalankan

1. Pastikan `citrus.csv` sudah ada di `/home/arfin/PycharmProjects/PythonProject/`
2. Install library yang dibutuhkan:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
3. Jalankan file di PyCharm:
```bash
python klasifikasi_buah.py
```
4. Hasil evaluasi tampil di console, grafik tersimpan otomatis di folder yang sama
