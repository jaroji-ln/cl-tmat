# Protokol Pelaksanaan: Continual Learning untuk Peramalan TMAT Gambut Tropis

Versi 5 · 15 November 2026 · dasar data: Taufik dkk. (2022), *Data in Brief* 41:107903

> **Aturan dokumen (sejak v4):** protokol ini tidak lagi memuat angka hasil yang diturunkan dari kode. Angka semacam itu hanya hidup di `results/` dan dirujuk dengan nama berkas. Ini mencegah protokol dan kode terus berselisih setiap kali pipeline dijalankan ulang.
>
> **Perubahan v5:** aturan 2.4 sudah diterapkan → pembingkaian benchmark (2.4.5); rencana diagnostik D1–D5 ditetapkan di 2.5 sebelum dijalankan; pemicu sadar-rezim ditangguhkan; perbandingan CL dipersempit ke Replay + EWC. Kalimat konteks 2.4 diedit secara redaksional (angka diganti rujukan); teks normatif 2.4 tidak berubah dari versi yang di-tag.
>
> **Perubahan v4:** aturan keputusan horizon ditetapkan di 2.4 sebelum hasil t+14 terlihat; angka galat koordinat, rasio degradasi, dan defisit Kubu Raya diganti rujukan ke berkas hasil.

> **Perubahan v3:** CHIRPS mulai 2018-01-01; batas informasi = t; skill terhadap persistensi menjadi metrik utama; rumus forgetting dibetulkan tandanya; uji signifikansi dipasangkan per sel stasiun × task; satu putaran seleksi hiperparameter di T1–T2; daftar keterbukaan ditambah hasil baseline T3–T5; galat koordinat dari skrip (269 m, 244 m).
>
> **Perubahan v2 dari versi 1:** jumlah hari T3 dikoreksi (1.415, bukan 1.215); koordinat stasiun sudah tersedia (Fase 0.3); segmen BRG3 2019 ditandai `is_suspect`; tabel drift lama dipensiunkan karena skripnya tidak tersimpan dan diganti angka yang dibangun ulang dari kode repo; Fase 3.3 ditambah prosedur penguncian pemicu dan pernyataan keterbukaan.

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

Artikel hanya mencantumkan koordinat BRG6 dan BRG18 (Specifications Table). Enam lainnya diekstraksi fotogrametrik dari Gambar 1 artikel (CC BY 4.0) dengan `src/extract_station_coords.py`, divalidasi terhadap dua stasiun yang diketahui (galat tercatat di `results/coord_validation.csv`; sejak 15 Nov 2026 memakai gambar beresolusi lebih tinggi), ketidakpastian sisa ±250 m. Berkas: `data/external/brg_station_coordinates.csv`.

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

CHIRPS harian lewat Google Earth Engine (`src/00_fetch_chirps.js`), interpolasi bilinear, **2018-01-01 s/d 2020-01-31** — lebih dari 90 hari sebelum data TMAT pertama (2018-04-08), agar jendela defisit 90 hari sudah lengkap sejak hari pertama. ERA5-Land opsional untuk variabel tambahan.

### 1.3 Rekayasa fitur

`src/02_build_features.py`. Target: TMAT pada **t+7**.

**Batas informasi: t (bukan t−1).** Prakiraan dibuat pada akhir hari t, setelah pembacaan hari t tersedia. Ini sesuai praktik operasional: stasiun TMAT BRG/SiTMAT bersifat telemetri dan melapor setiap jam, sehingga pembacaan hari itu sudah ada saat keputusan dibuat. Semua fitur lag dimulai dari t (lag 0), persistensi memakai TMAT t, dan hujan memakai hujan sampai hari t. Tidak ada informasi setelah t yang boleh masuk fitur.

| Kelompok | Fitur |
| --- | --- |
| Lag TMAT | t, t-1, t-2, t-3, t-5, t-7, t-14, t-21, t-30 |
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

Satu saja, konsisten: LSTM kecil atau TCN. Bukan Transformer.

**Seleksi hiperparameter — satu putaran, lalu dikunci.** Grid kecil yang ditulis sebelum dijalankan (mis. unit {32, 64}, lapis {1, 2}, dropout {0, 0,2}, lr {1e-3, 3e-3}, panjang jendela {14, 30}), validasi berbasis waktu **di dalam T1–T2 saja** (T1 + awal T2 untuk latih, akhir T2 untuk validasi). Anggaran seleksi yang setara diberikan juga ke Ridge (alpha) dan LightGBM, supaya perbandingan adil. Konfigurasi terpilih dikunci di `config/models.yaml` bersama tag pemicu. Naskah menyatakan terbuka bahwa putaran ini dilakukan setelah hasil baseline awal terlihat.

### 2.3 Bukti degradasi (Gambar 1c)

Tabel drift versi 1 (BRG3 5,84×, BRG6 2,53×, dst.) **dipensiunkan**: skrip pembuatnya tidak tersimpan sehingga tidak bisa direproduksi, dan angka BRG3-nya digerakkan segmen suspect.

Pengganti: rasio MAE kemarau 2019 / pra-kemarau, Ridge **panel**, suspect dikeluarkan, batas informasi t. Angkanya di `results/degradation_ratios.csv` dan dibangun ulang oleh skrip Gambar 1.

Yang dilaporkan apa adanya, berapa pun angkanya: degradasi saat kemarau **heterogen antar stasiun**, dan sebagian stasiun hampir tidak memburuk pada model panel. Itu diterima apa adanya dan menjadi bahan diskusi (mengapa model global sudah cukup untuk sebagian stasiun dan tidak untuk yang lain), bukan disembunyikan. Rasio per stasiun dilaporkan di lampiran sebagai pembanding, bukan di gambar utama.

---

### 2.4 Aturan keputusan horizon — ditetapkan 15 November 2026, sebelum hasil t+14 dijalankan

Konteks: pada t+7 tidak ada model realistis yang skill-nya terbukti di atas persistensi (`results/baseline_*.csv`). Aturan di bawah dikunci sebelum horizon lain dilihat, supaya pemilihan horizon tidak bisa dituduh mencari-cari hasil. Commit dan tag (`horizon-rule-locked`) sebelum eksperimen horizon dijalankan.

1. **Himpunan horizon dikunci sekarang: H = {3, 7, 14, 21, 30} hari.** Semuanya dijalankan dalam satu batch dengan pipeline yang sama, dan seluruhnya dilaporkan apa pun hasilnya. Tidak ada horizon yang ditambahkan setelah hasil terlihat.
2. **Seleksi hiperparameter per horizon** memakai prosedur 2.2: hanya fold latih T1–T2 (latih pada T1 + 70% awal fold latih T2, validasi pada 30% akhirnya; holdout T2 tidak disentuh). Grid 32 konfigurasi per keluarga model, ditulis di `config/hp_grid.yaml` sebelum dijalankan. LSTM diseleksi dengan satu seed (seed 0) — diungkap sebagai keterbatasan.
3. **Kriteria "ada skill" pada horizon h** — ketiganya harus terpenuhi:
   a. **Data yang dinilai: prakiraan online (prequential) pada T3–T5** — setiap prakiraan dibuat oleh model yang tersedia pada tanggal itu. T3–T5 adalah periode luar-sampel yang sama untuk semua kandidat, termasuk B2 yang dilatih di T1–T2. Holdout dengan model setelah task terakhir tetap dilaporkan sebagai metrik sekunder.
   b. **Kandidat: kelima model realistis** (B2 Ridge, B2 LightGBM, B5 Ridge, B5 LightGBM, B5 LSTM); B4 tidak termasuk. Tidak ada langkah "pilih yang terbaik". Setiap kandidat diuji: median skill per sel stasiun × task > 0, Wilcoxon berpasangan satu sisi terhadap persistensi, p dikoreksi **Holm atas 5 kandidat**, ambang 0,05. Kandidat lolos bila p-Holm < 0,05 **dan** skill online gabungannya ≥ +5%.
   c. **Oracle B3 dari keluarga model yang sama** dengan kandidat yang lolos punya skill holdout > 0 (tanpa syarat signifikansi). Oracle tidak punya versi online, jadi kriteria ini memakai holdout.
   Horizon h memenuhi kriteria bila setidaknya satu kandidat lolos a–c.
4. **Horizon utama = horizon TERPENDEK di H yang memenuhi kriteria**, bukan horizon dengan skill terbesar. Horizon terpendek paling bernilai operasional dan mencegah pemilihan puncak kurva. Horizon lain menjadi kurva skill pendukung (Fase 4.6).
5. **Bila tidak ada horizon yang memenuhi kriteria: tidak ada perpanjangan lagi.** Artikel dibingkai ulang sebagai studi benchmark: *persistensi sebagai baseline yang sulit dikalahkan untuk TMAT gambut, dan kapan pembelajaran adaptif sepadan*. Kontribusinya: kurva skill terhadap horizon, kegagalan retrain online selama kemarau T4, temuan kualitas data BRG3, dan protokol evaluasi. Pemicu sadar-rezim turun menjadi analisis sekunder. Setiap ide metode baru yang muncul setelah melihat hasil (misalnya gerbang yang kembali ke persistensi saat kemarau terdeteksi) wajib diberi label eksploratif pasca-hoc.
6. **Metrik kejadian** (F1 deteksi TMAT < −0,4 m, termasuk kejadian melewati ambang) dilaporkan untuk semua horizon sebagai metrik sekunder. Metrik ini **tidak boleh** dipakai untuk mengubah pilihan horizon utama.

### 2.5 Status keputusan dan diagnostik — ditetapkan 15 November 2026, sebelum dijalankan

**Keputusan 2.4 (final, tidak dibuka ulang):** tidak ada horizon dalam H yang memenuhi kriteria (`results/horizon_decision.yaml`). Artikel dibingkai sebagai benchmark sesuai 2.4.5.

**Mengapa diagnostik tetap wajib.** Klaim "persistensi sulit dikalahkan" baru kuat bila kegagalan model bukan artefak desain pipeline. Ada tiga dugaan artefak yang akan langsung diajukan reviewer:

1. Model memprediksi **level** TMAT, padahal pada horizon pendek hampir seluruh informasinya ada di level hari ini. Model yang memprediksi **perubahan** dari hari ini, dengan regularisasi menuju nol, tidak bisa jauh lebih buruk dari persistensi.
2. Normalisasi dibekukan dari T1. Selama kemarau T4, fitur mencapai |z| hingga 17, sehingga model linear berekstrapolasi liar. Normalisasi berjendela-mengembang yang hanya memakai data masa lalu tetap bebas kebocoran.
3. B5 hanya dilatih ulang dengan baris fold latih, padahal target holdout yang sudah teramati sebelum tanggal prakiraan sah dipakai.

**Diagnostik yang dijalankan (seluruhnya berlabel eksploratif pasca-hoc, dan TIDAK dapat membatalkan keputusan 2.4):**

| Kode | Perubahan tunggal terhadap pipeline terkunci | Menjawab |
|---|---|---|
| D1 | Target = TMAT(t+h) − TMAT(t); prakiraan = TMAT(t) + prediksi | Dugaan 1 |
| D2 | Normalisasi berjendela-mengembang, hanya data ≤ t | Dugaan 2 |
| D3 | D1 + D2 bersamaan | Interaksi |
| D4 | B5 dilatih ulang dengan semua baris yang targetnya sudah teramati sebelum tanggal prakiraan | Dugaan 3 |
| D5 | F1 kejadian **melewati** ambang −0,4 m (transisi dari atas ke bawah ambang dalam jendela h), bukan status di bawah ambang | Metrik T4 yang tidak informatif |

Ketentuan: kandidat, horizon H, grid hiperparameter, dan prosedur seleksi sama persis dengan versi terkunci. Hasil dilaporkan memakai tabel kriteria 2.4 yang sama, untuk semua kandidat dan horizon, apa pun hasilnya. Berkas `results/diagnostics_*.csv`.

**Cara membaca hasilnya di naskah:**
- Bila D1–D4 tetap kalah dari persistensi → klaim benchmark diperkuat: kegagalan bukan artefak parametrisasi, normalisasi, atau perlakuan B5.
- Bila salah satu mengalahkan persistensi → itu temuan eksploratif tentang **bagaimana** model harus dibangun untuk TMAT gambut, bukan pembatalan keputusan. Diuji konfirmatif pada data independen (lihat di bawah).

**Data konfirmatif.** Hipotesis yang lahir dari diagnostik ini (misalnya target berbentuk perubahan, atau skill pada rezim pembasahan ulang T5) diuji pada deret SiTMAT Riau yang dikumpulkan secara prospektif — data yang belum pernah dilihat saat hipotesis dirumuskan.

## FASE 3 — Metode continual learning

### 3.1 Protokol

Domain-incremental: T1 → T2 → T3 → T4 → T5. Setelah setiap task, evaluasi pada semua task yang sudah lewat → matriks R 5×5.

### 3.2 Metode pembanding

Minimal empat: Experience Replay, EWC, FSNet, dan satu lagi (latent replay, LwF, atau OneNet).

### 3.3 Pemicu sadar-rezim — DITANGGUHKAN (v5)

Setelah 2.4.5, pemicu tidak lagi menjadi kontribusi utama. Memutuskan *kapan* beradaptasi tidak bermakna selama tidak ada model adaptif yang mengalahkan persistensi. Kalibrasi dan tag `trigger-v1-locked` tidak dijalankan. Bagian ini dibuka kembali hanya bila diagnostik 2.5 menghasilkan model yang mengungguli persistensi; jika itu terjadi, pemicu berstatus eksploratif.

Perbandingan CL dipersempit ke **Experience Replay dan EWC** terhadap B5, dengan pertanyaan tunggal: apakah CL mengurangi keruntuhan skill pada T4 dibanding retrain berkala? Rancangan asli di bawah dipertahankan sebagai arsip.

#### Rancangan asli (arsip)

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

**Batas klaim yang jujur.** Tag hanya membuktikan konfigurasi terkunci sebelum eksperimen CL dijalankan, **bukan** bahwa kami tidak pernah melihat data T4/T5. Yang sudah dilihat sebelum penguncian: statistik defisit T4/T5, MAE Ridge, Gambar 1, dan **hasil baseline B1–B5 pada T3–T5**.

**Komponen galat (w3) hanya punya data online sejak T2,** karena T1 tidak memiliki model pendahulu. Kalibrasi w3 bertumpu pada T2 saja; nyatakan ini di naskah. Naskah harus menyatakannya terbuka. Pertahanan utama terhadap tuduhan penyetelan bukan tag, melainkan: (a) kurva sensitivitas ambang penuh (Fase 4.2), (b) ablasi yang memisahkan komponen domain dari komponen galat (Fase 4.1), dan (c) transfer lintas-pulau (Fase 4.3).

**Catatan kalibrasi:** rerata defisit hujan T2 di Kubu Raya negatif (angka di `results/`), artinya kemarau 2018 di Kalbar tidak defisit. Kalibrasi komponen defisit praktis hanya bertumpu pada Batanghari dan akhir T2. Laporkan ini, dan pertimbangkan kalibrasi per pulau atau normalisasi z per stasiun.

### 3.4 Metrik

| Metrik | Makna |
| --- | --- |
| **Skill terhadap persistensi (metrik utama)** | 1 − MAE_model / MAE_persistensi, per task dan rerata |
| MAE & RMSE akhir | Rerata semua task setelah task terakhir |
| Forgetting rerata | mean(galat akhir − galat terkecil historis) per task lama; positif = lupa |
| Backward / forward transfer | Pengaruh antar-task |
| Jumlah adaptasi | Berapa kali model diperbarui |
| Galat spesifik-kemarau | MAE pada T4 |
| F1 ambang −0,4 m | Deteksi TMAT < −0,4 m (PP 57/2016) |

### 3.5 Disiplin eksperimen

5 seed per konfigurasi (rerata ± sd); semua run dicatat ke `results/experiments.csv`.

**Uji signifikansi:** jangan menguji antar seed — dengan n = 5, Wilcoxon dua sisi tidak bisa mencapai p < 0,0625. Unit pasangan adalah **sel stasiun × task** (galat dirata-ratakan atas seed di tiap sel). Untuk dua metode: Wilcoxon signed-rank berpasangan per sel. Untuk banyak metode: Friedman lalu post-hoc Nemenyi atau Holm (Demšar, 2006, JMLR). Laporkan juga ukuran efek (median selisih skill), bukan hanya p.

---

## FASE 4 — Analisis dan ablasi

1. **Ablasi pemicu** — matikan tiap komponen sinyal satu per satu. Komponen domain (w1, w2) harus terbukti berkontribusi **di luar** komponen galat (w3). Kalau tidak, laporkan sebagai temuan.
2. **Sensitivitas ambang** — sapu ambang, gambar akurasi terhadap jumlah adaptasi untuk pemicu usulan dan jadwal tetap. Klaim keunggulan harus berlaku di sepanjang rentang ambang yang masuk akal, bukan satu titik.
3. **Transfer lintas-pulau** — kalibrasi/latih di Batanghari, uji di Kubu Raya, dan sebaliknya.
4. **Sensitivitas batas rezim** — geser batas rezim ±2 minggu; periksa apakah urutan metode berubah.
5. **Analisis kegagalan** — periksa langsung periode tempat semua metode gagal.
6. **Kurva skill terhadap horizon** — seluruh H = {3, 7, 14, 21, 30} sesuai aturan 2.4. Gambut berdinamika lambat, sehingga persistensi sangat kuat di horizon pendek; ruang nilai tambah model dan adaptasi kemungkinan lebih besar di horizon panjang. Kurva skill terhadap horizon menjawab langsung "kapan adaptasi sepadan".

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
| Tidak ada model yang skill-nya jelas di atas persistensi pada t+7 | **Nyata** (persistensi dengan data t sudah mengalahkan oracle t−1) | Terjadi pada t+7. Ditangani dengan aturan keputusan 2.4 yang dikunci sebelum horizon lain dilihat |
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
