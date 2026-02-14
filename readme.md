# ⚔️ AETHER BLADES: Pro AI Gesture Game

**Aether Blades** adalah demonstrasi teknologi *Computer Vision* yang mengubah webcam standar menjadi sensor gerak presisi tinggi. Pemain dapat berinteraksi dengan dunia virtual (menebas buah) hanya menggunakan gerakan tangan alami di udara tanpa alat tambahan.

---

## 🛠️ Panduan Persiapan Sistem (Langkah demi Langkah)

Proyek ini dibangun menggunakan bahasa pemrograman **Python**. Ikuti langkah-langkah di bawah ini untuk menjalankan game di komputer Anda.

### 1. Install Python (Jika belum ada)
Game ini memerlukan Python versi **3.8 atau yang lebih baru**.
*   **Windows**: 
    1. Download installer di [python.org](https://www.python.org/downloads/).
    2. **PENTING**: Saat menjalankan installer, pastikan centang kotak **"Add Python to PATH"** di bagian bawah sebelum klik *Install Now*.
*   **macOS**: Gunakan [Homebrew](https://brew.sh/): `brew install python`
*   **Linux**: Jalankan `sudo apt install python3 python3-pip`

### 2. Persiapkan Folder Proyek
Download kode ini atau clone melalui terminal:

git clone https://github.com/username_kamu/aether-blades.git
cd aether-blades


### 3. Buat Virtual Environment (Opsional tapi Disarankan)
Agar library game ini tidak bentrok dengan proyek lain di komputermu:
# Membuat environment
python -m venv venv

# Mengaktifkan environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

### 4. Install Library (Dependencies)
Game ini menggunakan tiga library utama: `opencv` (untuk kamera), `mediapipe` (untuk AI tangan), dan `numpy` (untuk matematika).
Jalankan perintah ini di terminal/cmd:
pip install opencv-python mediapipe numpy


---

## 🎮 Cara Menjalankan Game

Setelah semua library terinstall, jalankan perintah berikut:
python aether_blades.py

### Panduan Bermain:
1.  **Posisi**: Berdirilah sekitar 1-2 meter dari webcam hingga tanganmu terlihat jelas di layar.
2.  **Kalibrasi**: Pastikan cahaya ruangan terang (jangan membelakangi lampu/jendela).
3.  **Start**: Di menu utama, arahkan tangan ke lingkaran putih dan lakukan gerakan menebas dengan cepat.
4.  **Kontrol**:
    *   **Tangan Kiri**: Mengontrol Player 1.
    *   **Tangan Kanan**: Mengontrol Player 2.
    *   **Tombol R**: Reset/Rematch.
    *   **Tombol Q / ESC**: Keluar.

---

## 🧪 Detail Teknologi AI

*   **Hand Landmark Detection**: Menggunakan model *BlazePalm* dari MediaPipe yang mampu melacak 21 koordinat sendi tangan secara real-time.
*   **Adaptive Smoothing**: Algoritma khusus untuk meredam getaran (jitter) pada koordinat tangan agar gerakan pedang terasa mulus.
*   **Multithreading Audio**: Suara diproses di thread terpisah agar tidak menyebabkan lag pada frame rate game.
*   **Performance Scaling**: Secara otomatis menyesuaikan resolusi tracking untuk menjaga performa tetap stabil di 60 FPS pada kebanyakan laptop.

---

## ❓ Troubleshooting (Masalah Umum)

*   **Kamera Tidak Terbuka**: Pastikan tidak ada aplikasi lain (seperti Zoom atau Teams) yang sedang menggunakan webcam.
*   **Aplikasi Lag/Lambat**: Pastikan laptop Anda terhubung ke charger (mode *High Performance*). Resolusi kamera yang terlalu tinggi juga bisa berpengaruh.
*   **Error "ModuleNotFoundError"**: Berarti langkah nomor 4 di atas gagal atau Anda belum masuk ke Virtual Environment. Ulangi instalasi library.
*   **Suara Tidak Muncul**: Fitur suara saat ini dioptimalkan untuk Windows (`winsound`). Untuk macOS/Linux, game tetap berjalan normal namun tanpa suara.

---

## 👨‍💻 Kontribusi & Lisensi
Proyek ini bersifat open-source. Anda dipersilakan untuk melakukan *fork*, memodifikasi, dan menggunakan kode ini untuk portofolio pribadi.

**Dibuat oleh [Indrawan Arifianto]**