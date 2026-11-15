"""
Ekstraksi koordinat 8 stasiun BRG dari Gambar 1 Taufik dkk. (2022), Data in Brief 41:107903.

Artikel berlisensi CC BY 4.0. Gambar 1 dipakai sebagai masukan dengan atribusi.
Masukan : data/external/taufik2022_fig1.jpg  (versi resolusi tinggi, 2917 x 2046 px)
Keluaran: data/external/brg_station_coordinates.csv
          results/coord_validation.csv

Metode:
  1. Deteksi bingkai kedua panel peta dari garis piksel gelap yang panjang.
  2. Posisi garis koordinat = sentroid teks label sumbu (label rata-tengah terhadap tick).
     Pita label dicari relatif terhadap bingkai: bujur di atas bingkai, lintang di luar tepi
     bingkai, sehingga tidak bergantung pada potongan atau resolusi gambar.
  3. Kalibrasi linear per panel (derajat per piksel).
  4. Marker stasiun = komponen terhubung berbentuk bujur sangkar (fill > 0,80); batas ukuran
     diskalakan dengan lebar gambar.
  5. Validasi terhadap koordinat BRG6 dan BRG18 yang tercantum di Specifications Table artikel.
  6. Koreksi bias bujur rata-rata dari validasi, diterapkan pada stasiun hasil estimasi.
     Galat validasi diukur sebelum koreksi; enam stasiun lain tidak tervalidasi independen.

Konstanta piksel ditulis untuk lebar acuan 1194 px dan dikalikan SCALE = lebar / 1194.
Versi WebP 1194 x 844 dari ScienceDirect tidak dipakai: artefak kompresinya terbaca sebagai
label dan satu marker tidak terdeteksi.

Jalankan: python src/extract_station_coords.py [path_gambar]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
IMG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/external/taufik2022_fig1.jpg"
OUT_COORDS = ROOT / "data/external/brg_station_coordinates.csv"
OUT_VALID = ROOT / "results/coord_validation.csv"

DARK = 128          # ambang piksel gelap untuk bingkai dan label
MARKER_DARK = 90    # ambang lebih ketat untuk marker
MAP_ROWS = 0.68     # dua panel peta berada di atas fraksi tinggi ini
REFERENCE_WIDTH = 1194


def groups(vals, gap):
    out = []
    for v in vals:
        if out and v - out[-1][-1] <= gap:
            out[-1].append(v)
        else:
            out.append([v])
    return out


def centers(vals, gap, min_size=1):
    """Pusat tiap kelompok; kelompok lebih kecil dari min_size dianggap derau."""
    return [(g[0] + g[-1]) / 2 for g in groups(vals, gap) if len(g) >= min_size]


def dms(d, m, s):
    return d + m / 60 + s / 3600


def main():
    a = np.array(Image.open(IMG).convert("L"))
    H, W = a.shape
    scale = W / REFERENCE_WIDTH
    px = lambda v: max(1, round(v * scale))
    dark = a < DARK

    # --- 1. bingkai panel -------------------------------------------------
    upper = dark[: int(MAP_ROWS * H)]
    vcols = [x for x in range(W) if upper[:, x].sum() > 0.5 * upper.shape[0]]
    vlines = centers(vcols, gap=px(3))
    assert len(vlines) == 4, f"diharapkan 4 garis vertikal bingkai, dapat {vlines}"
    L_x0, L_x1, R_x0, R_x1 = vlines

    def frame_rows(x0, x1):
        band = upper[:, int(x0) + 2 : int(x1) - 1]
        rows = [y for y in range(band.shape[0]) if band[y].mean() > 0.9]
        lines = centers(rows, gap=px(3))
        return lines[0], lines[-1]

    L_top, L_bot = frame_rows(L_x0, L_x1)
    R_top, R_bot = frame_rows(R_x0, R_x1)

    # --- 2. sentroid label sumbu ------------------------------------------
    def lon_centroids(x0, x1, top):
        start = int(x0) + px(50)
        band = dark[: int(top) - px(2), start : int(x1)]
        cols = [start + x for x in range(band.shape[1]) if band[:, x].any()]
        return centers(cols, gap=px(12), min_size=px(5))

    def lat_centroids(x0, x1, top, bot):
        band = dark[int(top) : int(bot), int(x0) : int(x1)]
        rows = [int(top) + y for y in range(band.shape[0]) if band[y].any()]
        return centers(rows, gap=px(12), min_size=px(5))

    left_lon, right_lon = lon_centroids(L_x0, L_x1, L_top), lon_centroids(R_x0, R_x1, R_top)
    left_lat = lat_centroids(0, L_x0 - px(2), L_top, L_bot)
    right_lat = lat_centroids(R_x1 + px(3), W, R_top, R_bot)

    assert len(left_lon) == 3 and len(left_lat) == 3, (left_lon, left_lat)
    assert len(right_lon) == 2 and len(right_lat) == 2, (right_lon, right_lat)

    # --- 3. kalibrasi linear ------------------------------------------------
    cal = {
        "L": dict(
            lon=np.polyfit(left_lon, [103 + 40 / 60, 104.0, 104 + 20 / 60], 1),
            lat=np.polyfit(left_lat, [-1.0, -(1 + 20 / 60), -(1 + 40 / 60)], 1),
        ),
        "R": dict(
            lon=np.polyfit(right_lon, [109 + 20 / 60, 109 + 40 / 60], 1),
            lat=np.polyfit(right_lat, [0.0, -20 / 60], 1),
        ),
    }

    # --- 4. marker ----------------------------------------------------------
    lab, _ = ndimage.label(a < MARKER_DARK)
    markers = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if not (4 * scale <= h <= 12 * scale and 4 * scale <= w <= 12 * scale and abs(h - w) <= px(2)):
            continue
        fill = (lab[sl] == i).sum() / (h * w)
        cy, cx = (ys.start + ys.stop - 1) / 2, (xs.start + xs.stop - 1) / 2
        inside = (L_x0 < cx < L_x1 and L_top < cy < L_bot) or (R_x0 < cx < R_x1 and R_top < cy < R_bot)
        if fill > 0.80 and inside:
            markers.append((cx, cy))
    assert len(markers) == 8, f"diharapkan 8 marker, dapat {len(markers)}"

    # Penugasan label diverifikasi visual dengan overlay pada gambar resolusi tinggi.
    # Panel kiri, urut x: BRG3, BRG4, BRG6, BRG5. Panel kanan, urut x: BRG20, BRG17, BRG18, BRG19.
    left = sorted([m for m in markers if m[0] < L_x1])
    right = sorted([m for m in markers if m[0] > R_x0])
    names = dict(zip(["BRG3", "BRG4", "BRG6", "BRG5"], left))
    names.update(zip(["BRG20", "BRG17", "BRG18", "BRG19"], right))

    est = {}
    for st, (x, y) in names.items():
        c = cal["L"] if x < L_x1 else cal["R"]
        est[st] = (float(np.polyval(c["lon"], x)), float(np.polyval(c["lat"], y)))

    # --- 5. validasi --------------------------------------------------------
    truth = {
        "BRG6": (dms(103, 58, 9.47), -dms(1, 26, 34.12)),
        "BRG18": (dms(109, 31, 9.24), -dms(0, 16, 47.25)),
    }
    rows, dlon = [], []
    for st, (tlon, tlat) in truth.items():
        elon, elat = est[st]
        ex = (elon - tlon) * 111.32 * np.cos(np.radians(tlat))
        ey = (elat - tlat) * 110.57
        err_m = float(np.hypot(ex, ey) * 1000)
        dlon.append(tlon - elon)
        rows.append(dict(station=st, est_lon=elon, est_lat=elat, pub_lon=tlon, pub_lat=tlat, error_m=round(err_m, 1)))
    bias_lon = float(np.mean(dlon))
    OUT_VALID.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_VALID, index=False)

    # --- 6. keluaran final --------------------------------------------------
    out = []
    for st in ["BRG3", "BRG4", "BRG5", "BRG6", "BRG17", "BRG18", "BRG19", "BRG20"]:
        if st in truth:
            lon, lat, src = *truth[st], "published"
        else:
            lon, lat, src = est[st][0] + bias_lon, est[st][1], "estimated"
        region = "Batanghari, Jambi" if lon < 106 else "Kubu Raya, West Kalimantan"
        out.append(dict(station=st, longitude=round(lon, 5), latitude=round(lat, 5), source=src, region=region))
    OUT_COORDS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(OUT_COORDS, index=False)

    print(f"{IMG.name}: {W}x{H}, skala {scale:.3f}, 8 marker")
    print("Validasi:")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Bias bujur dikoreksi: {bias_lon:+.5f} deg")
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
