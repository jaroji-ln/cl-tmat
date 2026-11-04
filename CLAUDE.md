# cl-tmat — konteks riset untuk setiap sesi

Basis kode artikel riset untuk jurnal Q1/Q2 (Ecological Informatics, Environmental Modelling & Software, atau IEEE Access).

**Pertanyaan riset.** Metode continual learning mana yang paling efektif menjaga akurasi peramalan tinggi muka air tanah (TMAT) gambut tropis saat rezim hidrologi berubah, dan dapatkah pemicu adaptasi berbasis pengetahuan domain mengungguli jadwal adaptasi tetap?

**Kontribusi utama.** Pemicu adaptasi sadar-rezim yang memakai indikator kekeringan hidrologi (defisit hujan kumulatif 90 hari, laju penurunan TMAT 14 hari), bukan jadwal kalender.

## Tujuh aturan yang tidak boleh dilanggar

1. Semua pembagian data berbasis waktu, tidak pernah acak. Pembagian acak pada deret waktu adalah kebocoran data dan akan langsung ditolak reviewer.
2. Statistik normalisasi diambil dari task pertama (T1) saja, lalu dibekukan. Menormalkan dengan statistik seluruh data adalah kebocoran halus yang fatal bila ketahuan.
3. Buang baris yang nilai targetnya hasil imputasi. Interpolasi maksimal 5 hari berturut, tandai di kolom `is_imputed`.
4. Model panel, bukan per stasiun. ID stasiun sebagai fitur.
5. Setiap eksperimen dicatat ke `results/experiments.csv`: konfigurasi, seed, semua metrik, waktu jalan.
6. `data/raw/` tidak pernah di-commit. Masuk `.gitignore`.
7. Jangan pernah mengarang angka hasil. Kalau sebuah langkah gagal atau hasilnya mencurigakan, laporkan apa adanya.

## Fakta data terverifikasi

`data/raw/mmc1.xlsx` (Taufik dkk. 2022, Data in Brief 41:107903).

- Sheet `gwt` (3.929 baris; kolom `date`, `groundwater table (m)`, `station`, `peatland ` — nama kolom terakhir berspasi di belakang) dan `wrc` (110 baris, kurva retensi air, tidak dipakai untuk peramalan).
- 8 stasiun: BRG3–BRG6 di Batanghari, Jambi; BRG17–BRG20 di Kubu Raya, Kalimantan Barat. Rentang 2018-04-08 s.d. 2019-12-31.
- Satuan METER, negatif di bawah permukaan tanah. Teks artikel benar; Gambar 2 artikel keliru. Jangan konversi ke cm.
- Baris (NaN setelah reindex harian): BRG3 507, BRG4 265, BRG5 630 (86), BRG6 545, BRG17 461 (26), BRG18 264, BRG19 629 (94), BRG20 628 (188).
  BRG5 punya 85 NaN di berkas mentah plus 1 tanggal hilang (2019-03-22), sehingga 86 setelah reindex.
- Tanggal ganda: BRG5 2019-03-21 (−0,362 dan −0,316); digabung dengan rerata.
- Berkas ini tidak memuat hujan; hujan dari CHIRPS.
- Koordinat: `data/external/brg_station_coordinates.csv`. Hanya BRG6 dan BRG18 yang terpublikasi; enam lainnya estimasi fotogrametrik (±250 m). BRG19 berjarak ~66 m dari batas sel bujur dan ~166 m dari batas sel lintang CHIRPS, jadi wajib interpolasi bilinear, bukan nearest-cell.

## Rezim hidrologi (task continual learning)

| Label | Periode | Hari terpakai (8 stasiun), tabel brief |
|---|---|---|
| T1_transisi_2018 | 2018-04-08 – 2018-06-30 | 649 |
| T2_kemarau_2018 | 2018-07-01 – 2018-10-31 | 984 |
| T3_basah_201819 | 2018-11-01 – 2019-06-30 | 1.215 (**BELUM COCOK**: pipeline menghasilkan 1.415; menunggu konfirmasi peneliti) |
| T4_kemarau_2019 (El Niño) | 2019-07-01 – 2019-10-31 | 381 |
| T5_pemulihan_2019 | 2019-11-01 – 2019-12-31 | 136 |

T4 dan T5 tipis; karena itu model harus panel/global. Jangan buat model per stasiun.

## Bukti drift dari brief

Ridge, hanya fitur lag, tanpa hujan, horizon 7 hari, latih 2018, uji 2019.

| Stasiun | MAE pra-kemarau (m) | MAE kemarau 2019 (m) | Rasio |
|---|---|---|---|
| BRG3 | 0,051 | 0,298 | 5,84× |
| BRG6 | 0,057 | 0,144 | 2,53× |
| BRG5 | 0,059 | 0,117 | 1,98× |
| BRG20 | 0,053 | 0,088 | 1,66× |
| BRG19 | 0,043 | 0,059 | 1,37× |

Status reproduksi: lihat `results/drift_regression.csv` dan `tests/test_drift_regression.py`. Jangan menyesuaikan angka atau konfigurasi demi mencocokkan tabel.

## Konvensi pipeline

- Batas informasi: fitur baris tanggal t hanya memakai data sampai t−1; target adalah TMAT t+7.
- Kolom `task` ditentukan oleh **tanggal target**, sehingga target baris latih tidak pernah jatuh ke rezim uji. Pakai `src/splits.py` untuk semua pembagian.
- Baris dibuang bila target NaN/imputasi, bila TMAT t−1 hasil imputasi (titik ujung interpolasinya masih di masa depan), atau bila ada fitur NaN.
- `features.csv` belum dinormalisasi. Normalisasi (aturan 2) dikerjakan di sesi model dengan statistik T1.
- `rain_deficit_90d` = normal − jumlah hujan 90 hari (mm); positif berarti lebih kering dari normal.
- Normal hujan: klimatologi CHIRPS 1988–2017, seluruhnya sebelum periode studi. Hari kering: hujan < 1 mm.
- Laju penurunan TMAT 14 hari untuk pemicu belum dibuat; dihitung di sesi continual learning.

## Masalah terbuka (per 2026-11-04)

Jangan menganggap ini selesai sebelum peneliti memutuskan.

1. **T3:** pipeline 1.415 hari, tabel brief 1.215; `test_regime_day_counts_match_brief[T3...]` merah. T1, T2, T4, T5 cocok persis.
2. **Tabel drift tidak tereproduksi** dalam toleransi 0,02 m (selisih terbesar 0,045 m per stasiun, 0,105 m panel); `test_drift_table_is_reproduced_within_tolerance` merah. Konfigurasi asli tabel tidak diketahui.
3. **Segmen BRG3 mencurigakan:** lompatan −0,254 m pada 2019-07-10, lalu laju konstan −0,007 m/hari selama 45 hari sampai akhir deret 2019-08-27. Belum dikeluarkan dari data. BRG3 adalah bukti drift terkuat; perlu diperiksa sebelum dipakai sebagai klaim.
4. **`01_prepare_gwt.py` ditulis dari spesifikasi**, bukan versi teruji milik peneliti yang tidak tersedia di repositori. Bandingkan bila versi itu tersedia.
5. **Bilinear CHIRPS belum terverifikasi independen.** `00_fetch_chirps.js` sudah dijalankan peneliti dan keluarannya lolos validasi kelengkapan, tetapi perbedaannya dengan nearest-cell belum diuji langsung.
6. **Defisit 90 hari tertinggal dari kejadian.** Pada baris `features.csv`, porsi defisit > 0 per task: T1 0%, T2 57%, T3 34%, T4 79%, T5 93%. BRG3, penurunan terbesar di T4, justru rerata defisitnya −7 mm. Rerata T5 (195 mm) melebihi T4 (139 mm). Perlu dievaluasi bersama laju penurunan TMAT 14 hari, bukan sendiri.
7. **Ambang pemicu tidak bisa dikalibrasi dari T1**, karena defisit T1 tidak pernah positif. Ambang juga tidak boleh dipilih dengan melihat T4/T5.
8. **Statistik T1 membuat T4 jauh di luar rentang:** rerata z defisit T4 +4,0 (maks +7,0); 57% baris T4 punya fitur dengan |z| > 5; |z| terbesar 17,2 (`dry_spell_days`). Aturan 2 tetap berlaku; dampaknya harus ditangani di sesi model.
9. **Satu kalender rezim untuk dua provinsi kurang pas di Kubu Raya:** Oktober 2018 dan Oktober 2019 berlabel kemarau, tetapi hujannya sekitar 450–490 mm dan 340–390 mm.
