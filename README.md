# 📊 Klasifikasi Buah: Orange vs Grapefruit

## 📌 Deskripsi Proyek

Proyek ini bertujuan untuk membangun model Machine Learning yang dapat mengklasifikasikan buah menjadi dua kategori, yaitu **orange** dan **grapefruit**, berdasarkan karakteristik fisik dan warna buah.

Dataset yang digunakan berisi informasi mengenai diameter, berat, serta nilai warna RGB dari masing-masing buah.

---

## 📁 Deskripsi Dataset

Dataset memiliki atribut sebagai berikut:

| Fitur    | Deskripsi                   |
| -------- | --------------------------- |
| diameter | Diameter buah (cm)          |
| weight   | Berat buah (gram)           |
| red      | Nilai warna merah (0–255)   |
| green    | Nilai warna hijau (0–255)   |
| blue     | Nilai warna biru (0–255)    |
| name     | Label (orange / grapefruit) |

Target (label) pada dataset adalah kolom **name**, yang menunjukkan jenis buah.

---

## 🔍 Data Understanding (EDA)

Berdasarkan eksplorasi data:

* Distribusi diameter dan berat menunjukkan perbedaan antara orange dan grapefruit.
* Nilai RGB membantu dalam membedakan warna khas masing-masing buah.
* Data cenderung memiliki pola yang dapat dipisahkan (separable), sehingga cocok untuk algoritma klasifikasi.

---

## ⚙️ Data Preprocessing

Langkah-langkah preprocessing yang dilakukan:

1. Encoding label:

   * orange → 0
   * grapefruit → 1
2. Pemisahan fitur (X) dan label (y)
3. Pembagian data:

   * Training set: 80%
   * Testing set: 20%
4. Normalisasi data menggunakan StandardScaler (khusus untuk SVM dan Naive Bayes)

---

## 🤖 Model yang Digunakan

### 1. Decision Tree

* Tidak memerlukan normalisasi data
* Mudah diinterpretasikan
* Rentan terhadap overfitting

### 2. Naive Bayes

* Cepat dan efisien
* Mengasumsikan independensi antar fitur
* Cocok untuk dataset sederhana

### 3. Support Vector Machine (SVM)

* Cocok untuk data yang dapat dipisahkan secara linear
* Memiliki performa yang tinggi
* Lebih kompleks dibanding model lainnya

---

## 🏋️ Training Model

Setiap model dilatih menggunakan data training yang telah dipisahkan sebelumnya.

---

## 📈 Evaluasi Model

Model dievaluasi menggunakan beberapa metrik berikut:

* **Accuracy** → Mengukur tingkat akurasi prediksi
* **Confusion Matrix** → Menunjukkan jumlah prediksi benar dan salah
* **Classification Report** → Berisi precision, recall, dan f1-score

---

## 📊 Perbandingan Model

| Model         | Kelebihan        | Kekurangan               |
| ------------- | ---------------- | ------------------------ |
| Decision Tree | Mudah dipahami   | Rentan overfitting       |
| Naive Bayes   | Cepat dan ringan | Asumsi terlalu sederhana |
| SVM           | Akurasi tinggi   | Komputasi lebih berat    |

---

## 🏆 Hasil dan Analisis

Berdasarkan hasil pengujian:

* Ketiga model mampu melakukan klasifikasi dengan baik.
* Model **Support Vector Machine (SVM)** umumnya memberikan akurasi tertinggi.
* Hal ini dikarenakan dataset memiliki pola yang relatif dapat dipisahkan dengan jelas.

---

## 📌 Kesimpulan

* Dataset citrus dapat digunakan dengan baik untuk permasalahan klasifikasi biner.
* Fitur seperti diameter, weight, dan RGB memiliki pengaruh signifikan dalam membedakan jenis buah.
* Model terbaik dalam eksperimen ini adalah **Support Vector Machine (SVM)**.

---

## 🚀 Saran Pengembangan

* Menambahkan visualisasi data (histogram, scatter plot)
* Mencoba algoritma lain seperti Random Forest atau KNN
* Melakukan hyperparameter tuning untuk meningkatkan performa model
* Menggunakan cross-validation untuk evaluasi yang lebih stabil

---
