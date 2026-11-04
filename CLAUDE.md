# cl-tmat — konteks riset untuk setiap sesi

Protokol lengkap: `docs/protokol-riset-cl-tmat.md`.

Basis kode artikel riset untuk jurnal Q1/Q2 (Ecological Informatics, Environmental Modelling & Software, atau IEEE Access).

**Pertanyaan riset.** Metode continual learning mana yang paling efektif menjaga akurasi peramalan tinggi muka air tanah (TMAT) gambut tropis saat rezim hidrologi berubah, dan dapatkah pemicu adaptasi berbasis pengetahuan domain mengungguli jadwal adaptasi tetap?

**Kontribusi utama.** Pemicu adaptasi sadar-rezim yang memakai indikator kekeringan hidrologi (defisit hujan kumulatif 90 hari, laju penurunan TMAT 14 hari), bukan jadwal kalender.

## Aturan yang tidak boleh dilanggar

1. Semua pembagian data berbasis waktu, tidak pernah acak. Pembagian acak pada deret waktu adalah kebocoran data dan akan langsung ditolak reviewer.
2. Statistik normalisasi diambil dari task pertama (T1) saja, lalu dibekukan. Menormalkan dengan statistik seluruh data adalah kebocoran halus yang fatal bila ketahuan.
3. Buang baris yang nilai targetnya hasil imputasi. Interpolasi maksimal 5 hari berturut, tandai di kolom `is_imputed`.
4. Model panel, bukan per stasiun. ID stasiun sebagai fitur.
5. Setiap eksperimen dicatat ke `results/experiments.csv`: konfigurasi, seed, semua metrik, waktu jalan.
6. `data/raw/` tidak pernah di-commit. Masuk `.gitignore`.
7. Jangan pernah mengarang angka hasil. Kalau sebuah langkah gagal atau hasilnya mencurigakan, laporkan apa adanya.
8. Angka yang tidak punya skrip di repositori tidak boleh dipakai (ditetapkan peneliti, 2026-11-04). Angka kualitas data dibangun ulang oleh `src/81_data_audit.py` ke `results/data_audit.csv`.

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
| T3_basah_201819 | 2018-11-01 – 2019-06-30 | 1.415 (dikoreksi peneliti dari 1.215, 2026-11-04) |
| T4_kemarau_2019 (El Niño) | 2019-07-01 – 2019-10-31 | 381 (termasuk 49 hari suspect BRG3, jadi 332 tanpa suspect) |
| T5_pemulihan_2019 | 2019-11-01 – 2019-12-31 | 136 |

T4 dan T5 tipis; karena itu model harus panel/global. Jangan buat model per stasiun.

## Segmen suspect BRG3

`is_suspect` menandai BRG3 2019-07-10 s.d. 2019-08-27 (49 hari). Nilainya tetap ada di `gwt_clean.csv`, tetapi dikeluarkan dari semua fitur, target, dan metrik. Keputusan peneliti: hampir pasti bukan pengukuran, dan harus dilaporkan terbuka di bagian data artikel.

- 2019-07-10: lompatan −0,254 m tanpa kejadian hujan (0 mm pada 07-09 dan 07-10).
- 2019-07-14 s.d. 08-27 (45 hari): garis lurus −7,17 mm/hari, residual 0,28 mm, selisih harian hanya −0,007/−0,008, sd selisih 0,00037 (115× lebih kecil dari BRG3 sebelumnya, sd 0,0427), berakhir tepat di ujung rekaman. Kelengkungan nol, padahal resesi gambut cekung; ada 58 mm hujan tanpa respons.
- Aturan objektif (`straight_runs` di `01_prepare_gwt.py`: ≥14 hari, rentang selisih ≤2 mm) hanya menangkap segmen ini di seluruh dataset. Hasil: `results/straight_runs.csv`.
- Pembanding berjendela sama (44 selisih): sd terkecil stasiun lain 11–56× lebih besar. BRG6 2019-07-23 s.d. 08-26: sd 0,00428, residual linear 5,41 mm, kelengkungan melambat.
- 2019-07-10 s.d. 07-13 ikut ditandai karena berada pada level pascalompatan. Bukan bagian garis lurus (residual menjadi 1,2 mm bila disertakan).

`results/drift_regression.csv` memisahkan `with_suspect` dan `suspect_excluded` (yang dikutip).

## Bukti degradasi (protokol 2.3)

Tabel drift brief awal **dipensiunkan** (skripnya tidak tersimpan). Jangan dikutip. Sumber tunggal: `results/drift_regression.csv` dari `src/80_drift_regression.py`. Panel (c) Gambar 1 dan klaim utama memakai `panel` + `suspect_excluded`; `per_station` masuk lampiran; `with_suspect` hanya untuk mengukur dampak segmen.

## Konvensi pipeline

- Batas informasi: fitur baris tanggal t hanya memakai data sampai t−1; target adalah TMAT t+7.
- Kolom `task` ditentukan oleh **tanggal target**, sehingga target baris latih tidak pernah jatuh ke rezim uji. Pakai `src/splits.py` untuk semua pembagian.
- Nilai `is_suspect` diperlakukan sebagai NaN. Baris dibuang bila target NaN/imputasi, bila TMAT t−1 hasil imputasi (titik ujung interpolasinya masih di masa depan), atau bila ada fitur NaN.
- `features.csv` belum dinormalisasi. Normalisasi (aturan 2) dikerjakan di sesi model dengan statistik T1.
- `rain_deficit_90d` = normal − jumlah hujan 90 hari (mm); positif berarti lebih kering dari normal.
- Normal hujan: klimatologi CHIRPS 1988–2017, seluruhnya sebelum periode studi. Hari kering: hujan < 1 mm.
- Laju penurunan TMAT 14 hari untuk pemicu belum dibuat; dihitung di sesi continual learning.

## Penguncian pemicu (keputusan peneliti, 2026-11-04)

- `config/trigger.yaml` memuat bobot, ambang, dan jendela minimum antar-adaptasi. Nilainya dikalibrasi dari T1–T2 saja. Saat lengkap: `status: frozen`, commit, git tag beranotasi, cantumkan hash di artikel. Baru setelah itu eksperimen T4/T5 boleh dijalankan. Jangan membuat tag selama nilainya masih null.
- Tag hanya membuktikan konfigurasi terkunci sebelum eksperimen, bukan bahwa data T4/T5 belum pernah dilihat. Statistik deskriptif defisit dan TMAT T4/T5 serta hasil Ridge sudah dilihat di Fase 1, dan komentar skrip asli peneliti menyebut batas rezim ditentukan dari pola musim pada data. Tulis ini terbuka di artikel.
- Kalibrasi pada kekeringan sedang (T2), uji pada yang ekstrem (T4): nyatakan sebagai pilihan desain. Keterbatasan yang harus ditulis sendiri: hanya satu episode kekeringan parah.
- Laporkan seluruh kurva akurasi terhadap jumlah adaptasi, bukan hanya ambang terpilih (sapuan Fase 4.2, transfer lintas-pulau Fase 4.3).
- Galat prakiraan bergerak 14 hari paling rawan dituduh penyetelan tersembunyi. Ablasi Fase 4.1 harus menunjukkan kontribusi komponen domain di luar umpan balik galat; bila tidak, laporkan sebagai temuan.

## Masalah terbuka (per 2026-11-04)

Jangan menganggap ini selesai sebelum peneliti memutuskan.

1. **Galat fotogrametri koordinat (273 m, 240 m, ±250 m) belum punya skrip di repositori**; kena aturan 8 bila dikutip di naskah.
2. **Bilinear CHIRPS belum terverifikasi independen.** `00_fetch_chirps.js` sudah dijalankan peneliti dan keluarannya lolos validasi kelengkapan, tetapi perbedaannya dengan nearest-cell belum diuji langsung.
3. **Defisit 90 hari tertinggal dari kejadian.** Pada baris `features.csv`, porsi defisit > 0 per task: T1 0%, T2 57%, T3 34%, T4 82%, T5 93%. Rerata T5 (195 mm) melebihi T4 (168 mm). Perlu dievaluasi bersama laju penurunan TMAT 14 hari, bukan sendiri.
4. **Ambang pemicu tidak bisa dikalibrasi dari T1**, karena defisit T1 tidak pernah positif. Ambang juga tidak boleh dipilih dengan melihat T4/T5.
5. **Statistik T1 membuat T4 jauh di luar rentang:** rerata z defisit T4 +4,4 (maks +7,0); 63% baris T4 punya fitur dengan |z| > 5; |z| terbesar 17,2 (`dry_spell_days`). Aturan 2 tetap berlaku; dampaknya harus ditangani di sesi model.
6. **Satu kalender rezim untuk dua provinsi kurang pas di Kubu Raya:** Oktober 2018 dan Oktober 2019 berlabel kemarau, tetapi hujannya sekitar 450–490 mm dan 340–390 mm.
