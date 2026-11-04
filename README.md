# cl-tmat

Peramalan tinggi muka air tanah (TMAT) gambut tropis dengan continual learning, dan pemicu adaptasi sadar-rezim berbasis indikator kekeringan hidrologi. Konteks riset dan aturan kerja ada di [CLAUDE.md](CLAUDE.md).

Status saat ini: Fase 0–1. Metode baseline dan continual learning belum ada di repositori ini.

## Menjalankan pipeline

```bash
pip install -r requirements.txt
# Letakkan mmc1.xlsx (Taufik dkk. 2022, Data in Brief 41:107903) di data/raw/. Berkas ini tidak di-commit.
cd src
python 01_prepare_gwt.py        # -> data/processed/gwt_clean.csv dan results/straight_runs.csv
python 02_build_features.py     # -> data/processed/features.csv; butuh dua berkas CHIRPS di bawah
python 80_drift_regression.py   # -> results/drift_regression.csv dan results/experiments.csv
python 90_figure1.py            # -> figures/fig1_tmat_regimes_drift.{pdf,png}
cd .. && python -m pytest tests
```

Skrip berhenti dengan pesan jelas bila masukannya belum ada; tidak ada data sintetis. Keluaran skrip deterministik.

`80_drift_regression.py` keluar dengan kode 2 bila tabel drift tidak tereproduksi.

## Mengambil hujan CHIRPS

1. Buka <https://code.earthengine.google.com> dengan akun yang terdaftar di Earth Engine.
2. Salin isi `src/00_fetch_chirps.js` ke skrip baru, lalu klik **Run**.
3. Di panel Console, pastikan jumlah citra harian bernilai 761.
4. Di tab **Tasks**, klik **Run** pada `chirps_stations` dan `chirps_climatology`. Keduanya tersimpan di Google Drive sebagai CSV.
5. Pindahkan hasilnya ke:
   - `data/external/chirps_stations.csv` (`station,date,precip_mm`, 8 × 761 = 6.088 baris)
   - `data/external/chirps_climatology.csv` (`station,doy,precip_clim_mm`, 8 × 366 = 2.928 baris)
6. Jalankan `python src/02_build_features.py`. Skrip memeriksa kelengkapan kalender dan rentang kedua berkas.

Rentang harian dimulai 2018-01-01, karena jendela 90 hari untuk hari pertama TMAT (2018-04-08) dimulai 2018-01-08. Klimatologi memakai 1988–2017, seluruhnya sebelum periode studi, sebagai "normal" untuk defisit hujan. Pengambilan sampel bilinear dilakukan pada skala 100 m; pada skala asli CHIRPS hasilnya sama dengan nearest-cell.

## Struktur

```
data/raw/        data mentah, tidak di-commit
data/external/   koordinat stasiun dan ekspor CHIRPS
data/processed/  keluaran pipeline, dibuat ulang oleh skrip
src/             skrip pipeline, common.py (konstanta), splits.py (pembagian temporal)
results/         experiments.csv dan hasil uji regresi
config/          konfigurasi pemicu, dikunci dengan git tag sebelum eksperimen T4/T5
figures/         gambar artikel
paper/           naskah
tests/           pytest
```
