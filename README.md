# Indonesian Typo Correction with IndoBART

Proyek ini adalah implementasi Automatic Spelling Correction (Koreksi Ejaan Otomatis) untuk Bahasa Indonesia. Model ini dilatih (fine-tuned) menggunakan arsitektur IndoBART v2 untuk memperbaiki kesalahan penulisan kata (typo) non-word error.

Proyek ini dikembangkan sebagai bagian dari Tugas Akhir NLP.

---

# 📋 Fitur Utama

- Fine-tuning IndoBART: Menggunakan model pre-trained indobenchmark/indobart-v2 (berbasis mBART) yang dilatih ulang dengan dataset spesifik.

- Word-Level Correction: Model dilatih untuk memetakan kata typo menjadi kata baku.

- Batch Sentence Processing: Script inferensi mampu menangani input kalimat dengan memecahnya per kata, memperbaikinya secara paralel, dan menggabungkannya kembali.

- Interactive Testing Tool: Menu interaktif untuk:

    - Generate soal typo acak dari dataset untuk menguji performa model.

    - Input manual dari pengguna.

---

# 🧠 Model & Dataset

Arsitektur Model

  - Base Model: IndoBART v2

- Arsitektur Dasar: mBART (Multilingual Denoising Autoencoder) dengan arsitektur Sequence-to-Sequence (Encoder-Decoder).

- Kenapa IndoBART? Model ini telah dilatih (pre-trained) dengan dataset masif Bahasa Indonesia, Jawa, dan Sunda, sehingga memiliki pemahaman morfologi bahasa lokal yang lebih baik dibandingkan mBART standar.

Dataset

- Nama: Saltik (Salah Ketik)

- Deskripsi: Dataset benchmark untuk koreksi kesalahan kata non-word dalam Bahasa Indonesia.

- Ukuran: Terdiri dari 58,532 variasi kesalahan ketik yang dihasilkan dari 3,000 kata dasar populer.

---

# 🛠️ Instalasi

Pastikan Anda telah menginstal Python (disarankan versi 3.8 ke atas) dan memiliki GPU (CUDA) untuk performa training yang optimal.

## Clone Repository ini:
```bash
git clone [https://github.com/username-anda/nama-repo-anda.git](https://github.com/username-anda/nama-repo-anda.git)
cd nama-repo-anda
```

## Install Library yang Dibutuhkan:
```bash
pip install torch transformers pandas numpy scikit-learn datasets evaluate sacrebleu nltk
```

(Disarankan menggunakan Virtual Environment)

---

# 🚀 Cara Penggunaan

1. Training Model (train_model.py)

Script ini digunakan untuk melatih model dari awal menggunakan dataset saltik.json.

- Pastikan file dataset berada di dataset/saltik.json.

- Script akan melakukan flattening data JSON menjadi format tabular.

- Training berjalan selama 5 Epoch dengan Batch Size 8.

- Model otomatis disimpan di folder indobart-correction-saltik/.

Cara Menjalankan:
```bash
python train_model.py
```

2. Menjalankan Aplikasi / Pengujian (run_model.py)

Setelah training selesai, gunakan script ini untuk mencoba model. Script ini memuat model yang tersimpan dan menyediakan antarmuka CLI sederhana.

Fitur:

- Generate Kalimat Random: Mengambil kata-kata acak dari dataset Saltik untuk melihat apakah model mampu mengembalikan ke bentuk aslinya.

- Input Manual: Ketik kalimat Anda sendiri (misal: "sya mkan nsi") dan lihat hasil koreksinya.

Cara Menjalankan:
```bash
python run_model.py
```

---

# 📂 Struktur Folder

```bash
├── dataset/
│   └── saltik.json              # Dataset sumber
├── indobart-correction-saltik/  # (Output) Model hasil training
├── checkpoints_saltik/          # (Output) Checkpoint training per epoch
├── train_model.py               # Script untuk melatih model
├── run_model.py                 # Script untuk menjalankan aplikasi
└── README.md                    # Dokumentasi proyek
```

---

# ⚠️ Catatan Penting

- Ukuran Model: Folder indobart-correction-saltik berisi file pytorch_model.bin yang berukuran besar (>600MB). Jika Anda meng-clone repo ini, Anda mungkin perlu mengunduh modelnya secara terpisah atau melatih ulang sendiri jika folder tersebut tidak disertakan karena batas ukuran GitHub (kecuali menggunakan Git LFS).

- Karakteristik Model: Model ini dilatih pada level kata. Jika Anda memasukkan kalimat, script run_model.py akan memecah kalimat tersebut berdasarkan spasi dan memperbaiki setiap kata secara individual.

---

# 🤝 Credits

- Terima kasih kepada IndoBenchmark untuk pre-trained model IndoBART.

- Terima kasih kepada pembuat dataset Saltik.

Created by Kelompok 1 NLP
