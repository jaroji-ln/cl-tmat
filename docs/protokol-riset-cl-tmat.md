# Protokol Pelaksanaan: Continual Learning untuk Peramalan TMAT Gambut Tropis

Versi 2 · 4 November 2026 · dasar data: Taufik dkk. (2022), *Data in Brief* 41:107903

> **Perubahan dari versi 1:** jumlah hari T3 dikoreksi (1.415, bukan 1.215); koordinat stasiun sudah tersedia (Fase 0.3); segmen BRG3 2019 ditandai `is_suspect`; tabel drift lama dipensiunkan karena skripnya tidak tersimpan dan diganti angka yang dibangun ulang dari kode repo; Fase 3.3 ditambah prosedur penguncian pemicu dan pernyataan keterbukaan.

---

## Ringkasan rencana

| | |
| --- | --- |
| Pertanyaan inti | Metode continual learning mana yang paling efektif menjaga akurasi peramalan TMAT saat rezim hidrologi berubah, dan dapatkah pemicu adaptasi berbasis pengetahuan domain mengungguli jadwal tetap pada anggaran adaptasi yang sama? |
| Kontribusi utama | Pemicu adaptasi sadar-rezim yang memakai indikator kekeringan hidrologi, bukan jadwal kalender |
| Data | 8 stasiun TMAT harian (Jambi + Kalbar, 2018–2019) + curah hujan CHIRPS |
| Durasi realistis | 16–20 minggu paruh waktu |
| Target | Q1: Ecological Informatics, Environmental Modelling & Software. Q1/Q2: IEEE Access. Cadangan: Hydrology / Water (MDPI) |

---

## FASE 0 — Persiapan

### 0.1 Repositori

Repositori `cl-tmat` di root. Struktur: `data/{raw,external,processed}/`, `src/`, `config/`, `results/`, `figures/`, `paper/`, `tests/`, `docs/`. Aturan metodologis yang mengikat ada di `CLAUDE.md`.

### 0.2 Data mentah

`data/raw/mmc1.xlsx` (tidak di-commit). README memuat DOI dataset asli dan cara menempatkannya.

### 0.3 Koordinat stasiun — SELESAI

Artikel hanya mencantumkan koordinat BRG6 dan BRG18 (Specifications Table). Enam lainnya diekstraksi fotogrametrik dari Gambar 1 artikel, divalidasi terhadap dua stasiun yang diketahui (galat 273 m dan 240 m), ketidakpastian sisa ±250 m. Berkas: `data/external/brg_station_coordinates.csv`.

BRG19 berada dekat sudut sel CHIRPS (±66 m dan ±166 m dari batas), jadi ekstraksi hujan memakai **interpolasi bilinear**, bukan nearest-cell.

Tetap hubungi penulis korespondensi (Muh Taufik, IPB): konfirmasi koordinat enam stasiun dan status segmen BRG3 2019 (lihat 1.1). Ini juga pembuka kolaborasi.

---

## FASE 1 — Penyiapan data

### 1.1 Bersihkan deret TMAT

`src/01_prepare_gwt.py`:

- menggabung tanggal ganda BRG5 (2019-03-21) dengan rerata;
- mengindeks ulang ke kalender harian penuh per stasiun;
- menginterpolasi rumpang **maksimal 5 hari** (`limit_area="inside"`), menandai `is_imputed`;
- memberi label rezim;
- menandai `is_suspect` pada segmen BRG3 **2019-07-10 s/d 2019-08-27 (49 hari)**.

Keluaran: `data/processed/gwt_clean.csv`.

| Rezim | Periode | Hari terpakai (8 stasiun) |
| --- | --- | --- |
| T1 transisi 2018 | 8 Apr – 30 Jun 2018 | 649 |
| T2 kemarau 2018 | 1 Jul – 31 Okt 2018 | 984 |
| T3 basah 2018/19 | 1 Nov 2018 – 30 Jun 2019 | **1.415** |
| T4 kemarau 2019 (El Niño) | 1 Jul – 31 Okt 2019 | 381 |
| T5 pemulihan | 1 Nov – 31 Des 2019 | 136 |

**Segmen BRG3 suspect.** Mulai 2019-07-14 selama 45 hari, selisih hariannya hanya −0,007 dan −0,008 m; residual terhadap garis lurus 0,28 mm; sekitar 115× lebih mulus dari BRG3 di luar segmen dan 11–56× lebih mulus dari runtun mana pun dengan panjang jendela yang sama. Kurva resesinya linear (resesi gambut asli melambat, seperti BRG6 pada periode yang sama), dan 58 mm hujan selama segmen tidak menimbulkan respons. Penanda dimulai 2019-07-10 karena ada lompatan −0,254 m tanpa hujan pada tanggal itu dan nilai 07-10 sudah di level pascalompatan. Aturan objektif (≥14 hari, rentang selisih ≤2 mm) hanya menangkap segmen ini di seluruh dataset; hasilnya di `results/straight_runs.csv`.

Perlakuan: nilai tetap ada di `gwt_clean.csv`, tetapi **tidak masuk fitur, target, maupun metrik**. Dilaporkan terbuka di bagian data naskah.

**Konsekuensi desain:** T4 dan T5 tipis — setelah segmen suspect dikeluarkan, T4 tinggal 216 baris fitur dan BRG3 hanya 9 baris di T4. Model **harus panel (global)** dengan ID stasiun sebagai fitur. Pertimbangkan menggabungkan T5 ke T4 bila hasil tidak stabil; putuskan dari hasil, dan catat keputusannya.

**Keterbukaan yang wajib ditulis di naskah:** batas rezim ditentukan dari pola musim yang terlihat pada data, bukan dari indeks eksternal. Sebutkan ini apa adanya, atau — lebih kuat — tunjukkan bahwa batasnya selaras dengan kalender musim BMKG / indeks ENSO, lalu uji sensitivitas terhadap pergeseran batas ±2 minggu.

### 1.2 Curah hujan

CHIRPS harian lewat Google Earth Engine (`src/00_fetch_chirps.js`), interpolasi bilinear, **2018-03-01 s/d 2020-01-31** — 5 minggu lebih awal dari data TMAT agar fitur kumulatif tersedia sejak hari pertama. ERA5-Land opsional untuk variabel tambahan.

### 1.3 Rekayasa fitur

`src/02_build_features.py`. Target: TMAT pada **t+7**.

| Kelompok | Fitur |
| --- | --- |
| Lag TMAT | t-1, t-2, t-3, t-5, t-7, t-14, t-21, t-30 |
| Dinamika TMAT | selisih 1 dan 7 hari; rerata bergerak 7 dan 30 hari |
| Hujan | harian; kumulatif 3, 7, 14, 30, 60 hari |
| Kekeringan | hari berturut tanpa hujan; defisit hujan kumulatif 90 hari terhadap normal |
| Musiman | sin/cos hari-dalam-tahun |
| Spasial | ID stasiun (one-hot), ID lahan gambut |

Keluaran saat ini: `data/processed/features.csv`, 3.004 baris.

### 1.4 Aturan pembagian data

- Semua pembagian berbasis waktu, tidak pernah acak.
- Buang baris yang targetnya hasil imputasi atau suspect.
- Normalisasi memakai statistik T1 saja, lalu dibekukan.

---

## FASE 2 — Baseline

### 2.1 Baseline wajib

| # | Baseline | Fungsi |
| --- | --- | --- |
| B1 | Persistensi (TMAT t+7 = TMAT t) | Batas bawah |
| B2 | Ridge / LightGBM statis, latih di T1–T2 | Praktik saat ini |
| B3 | Joint training semua task | Batas atas oracle |
| B4 | Fine-tuning naif berurutan | Menunjukkan forgetting |
| B5 | Retrain penuh berkala (tiap 30 hari) | Pembanding paling umum |

### 2.2 Tulang punggung DL

Satu saja, konsisten: LSTM kecil (2 lapis, 64 unit) atau TCN. Bukan Transformer.

### 2.3 Bukti degradasi (Gambar 1c)

Tabel drift versi 1 (BRG3 5,84×, BRG6 2,53×, dst.) **dipensiunkan**: skrip pembuatnya tidak tersimpan sehingga tidak bisa direproduksi, dan angka BRG3-nya digerakkan segmen suspect.

Pengganti: Ridge **panel**, suspect dikeluarkan, dibangun ulang dari kode repo:

| Stasiun | Rasio MAE kemarau 2019 / pra-kemarau |
| --- | --- |
| BRG6 | 2,94× |
| BRG5 | 2,71× |
| BRG3 (9 baris) | 2,16× |
| BRG19 | 1,31× |
| BRG20 | 1,08× |

Semua rasio > 1: degradasi saat kemarau nyata, tetapi **heterogen antar stasiun**. BRG20 hampir tidak memburuk pada model panel. Itu diterima apa adanya dan menjadi bahan diskusi (mengapa model global sudah cukup untuk sebagian stasiun dan tidak untuk yang lain), bukan disembunyikan. Rasio per stasiun dilaporkan di lampiran sebagai pembanding, bukan di gambar utama.

---

## FASE 3 — Metode continual learning

### 3.1 Protokol

Domain-incremental: T1 → T2 → T3 → T4 → T5. Setelah setiap task, evaluasi pada semua task yang sudah lewat → matriks R 5×5.

### 3.2 Metode pembanding

Minimal empat: Experience Replay, EWC, FSNet, dan satu lagi (latent replay, LwF, atau OneNet).

### 3.3 Kontribusi utama — pemicu sadar-rezim

```
sinyal_pemicu = w1 · z(defisit_hujan_90hari)
              + w2 · z(laju_penurunan_TMAT_14hari)
              + w3 · z(galat_prakiraan_bergerak_14hari)

jika sinyal_pemicu > ambang DAN hari_sejak_adaptasi_terakhir > minimum:
    jalankan adaptasi
```

Dibandingkan terhadap jadwal tetap pada **anggaran adaptasi yang sama**.

**Urutan wajib:**

1. Bangun baseline (Fase 2) — komponen galat prakiraan membutuhkannya.
2. Kalibrasi w1–w3, ambang, dan jarak minimum **hanya dari T1–T2**.
3. Tulis ke `config/trigger.yaml`, commit, beri **git tag** (`trigger-v1-locked`).
4. Baru jalankan eksperimen T3–T5. Setiap perubahan konfigurasi setelah tag dicatat sebagai versi baru dan dilaporkan.

**Batas klaim yang jujur.** Tag hanya membuktikan konfigurasi terkunci sebelum eksperimen CL dijalankan, **bukan** bahwa kami tidak pernah melihat data T4/T5. Statistik defisit T4/T5, MAE Ridge, dan Gambar 1 sudah dilihat sebelum penguncian. Naskah harus menyatakannya terbuka. Pertahanan utama terhadap tuduhan penyetelan bukan tag, melainkan: (a) kurva sensitivitas ambang penuh (Fase 4.2), (b) ablasi yang memisahkan komponen domain dari komponen galat (Fase 4.1), dan (c) transfer lintas-pulau (Fase 4.3).

**Catatan kalibrasi:** rerata defisit hujan T2 di Kubu Raya negatif (sekitar −44 s/d −67 mm), artinya kemarau 2018 di Kalbar tidak defisit. Kalibrasi komponen defisit praktis hanya bertumpu pada Batanghari dan akhir T2. Laporkan ini, dan pertimbangkan kalibrasi per pulau atau normalisasi z per stasiun.

### 3.4 Metrik

| Metrik | Makna |
| --- | --- |
| MAE & RMSE akhir | Rerata semua task setelah task terakhir |
| Forgetting rerata | mean(max galat historis − galat akhir) per task lama |
| Backward / forward transfer | Pengaruh antar-task |
| Jumlah adaptasi | Berapa kali model diperbarui |
| Galat spesifik-kemarau | MAE pada T4 |
| F1 ambang −0,4 m | Deteksi TMAT < −0,4 m (PP 57/2016) |

### 3.5 Disiplin eksperimen

5 seed per konfigurasi (rerata ± sd); semua run dicatat ke `results/experiments.csv`; uji Wilcoxon signed-rank antar metode.

---

## FASE 4 — Analisis dan ablasi

1. **Ablasi pemicu** — matikan tiap komponen sinyal satu per satu. Komponen domain (w1, w2) harus terbukti berkontribusi **di luar** komponen galat (w3). Kalau tidak, laporkan sebagai temuan.
2. **Sensitivitas ambang** — sapu ambang, gambar akurasi terhadap jumlah adaptasi untuk pemicu usulan dan jadwal tetap. Klaim keunggulan harus berlaku di sepanjang rentang ambang yang masuk akal, bukan satu titik.
3. **Transfer lintas-pulau** — kalibrasi/latih di Batanghari, uji di Kubu Raya, dan sebaliknya.
4. **Sensitivitas batas rezim** — geser batas rezim ±2 minggu; periksa apakah urutan metode berubah.
5. **Analisis kegagalan** — periksa langsung periode tempat semua metode gagal.

### Gambar

| # | Isi |
| --- | --- |
| 1 | (a) deret TMAT 8 stasiun dengan rezim diarsir, segmen suspect bertitik; (b) hujan; (c) rasio degradasi Ridge panel, suspect dikeluarkan |
| 2 | Diagram alur kerangka |
| 3 | Matriks forgetting antar metode |
| 4 | Kurva trade-off akurasi vs jumlah adaptasi |
| 5 | Prakiraan vs aktual selama kemarau 2019 |
| 6 | Hasil ablasi |

---

## FASE 5 — Penulisan

Urutan: Metode → Hasil → Diskusi → Pendahuluan → Abstrak.

### Daftar periksa sebelum submit

- [ ] Kode dirilis dengan README dan lingkungan yang bisa direproduksi
- [ ] Semua hiperparameter di lampiran
- [ ] Pembagian berbasis waktu dinyatakan eksplisit
- [ ] Hash commit/tag konfigurasi pemicu dicantumkan
- [ ] Keterbukaan: batas rezim dari pola data; data T4/T5 sudah dilihat sebelum penguncian
- [ ] Keterbatasan: 21 bulan, satu episode kekeringan parah, hujan satelit, enam koordinat hasil estimasi peta (±250 m), segmen BRG3 suspect, heterogenitas degradasi antar stasiun
- [ ] Dataset asli disitasi; temuan kualitas data dilaporkan
- [ ] Cek plagiarisme mandiri

Jangan kirim ke Journal of Hydrology.

---

## Risiko

| Risiko | Kemungkinan | Penanganan |
| --- | --- | --- |
| Metode CL tidak mengungguli retrain berkala | Nyata | Bingkai ulang: "kapan CL sepadan dan kapan tidak" |
| Komponen domain pemicu tidak berkontribusi | Sedang | Laporkan jujur dari ablasi; tetap layak terbit |
| Data T4 terlalu tipis untuk hasil stabil | Sedang–tinggi | Gabungkan T4–T5, perbanyak seed, turunkan target ke Q2 |
| Reviewer mempersoalkan penyetelan pada T4 | Sedang | Keterbukaan + kurva sensitivitas penuh + transfer lintas-pulau |
| Ditolak di Q1 | Tinggi (normal) | Perbaiki, turun ke venue berikutnya |

---

## Jadwal

| Minggu | Kegiatan | Luaran |
| --- | --- | --- |
| 1–3 | Repositori, data, hujan, fitur | Selesai sebagian: `gwt_clean.csv`, `features.csv`, Gambar 1 |
| 4–6 | Baseline B1–B5 | Tabel baseline |
| 7 | Kalibrasi pemicu T1–T2, tag | `trigger-v1-locked` |
| 8–10 | Metode CL + pemicu | Hasil perbandingan |
| 11–12 | Ablasi, sensitivitas, transfer | Semua gambar |
| 13–16 | Penulisan | Naskah |
| 17 | Perapian | Siap submit |

Cadangan 4 minggu.
