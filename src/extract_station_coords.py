"""
Ekstraksi koordinat 8 stasiun BRG dari Gambar 1 Taufik dkk. (2022), Data in Brief 41:107903.

Artikel berlisensi CC BY 4.0. Gambar 1 dipakai sebagai masukan dengan atribusi.
Masukan : data/external/taufik2022_fig1.png  (1194 x 844 px, versi yang diunduh dari ScienceDirect)
Keluaran: data/external/brg_station_coordinates.csv
          results/coord_validation.csv

Metode:
  1. Deteksi bingkai panel dari proyeksi piksel gelap.
  2. Posisi garis koordinat = sentroid teks label sumbu (label rata-tengah terhadap tick).
  3. Kalibrasi linear per panel (derajat per piksel).
  4. Marker stasiun = komponen terhubung berbentuk bujur sangkar (fill > 0,80).
  5. Validasi terhadap koordinat BRG6 dan BRG18 yang tercantum di Specifications Table artikel.
  6. Koreksi bias bujur rata-rata dari validasi, diterapkan pada stasiun hasil estimasi.

Jalankan: python src/extract_station_coords.py [path_gambar]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

IMG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/external/taufik2022_fig1.png")
OUT_COORDS = Path("data/external/brg_station_coordinates.csv")
OUT_VALID = Path("results/coord_validation.csv")

DARK = 128          # ambang piksel gelap untuk bingkai dan label
MARKER_DARK = 90    # ambang lebih ketat untuk marker
TOP_ROWS = 560      # dua panel peta berada di atas baris ini


def clusters(vals, gap=12):
    out = []
    for v in vals:
        if out and v - out[-1][-1] <= gap:
            out[-1].append(v)
        else:
            out.append([v])
    return [(c[0] + c[-1]) / 2 for c in out]


def dms(d, m, s):
    return d + m / 60 + s / 3600


def main():
    a = np.array(Image.open(IMG).convert("L"))
    H, W = a.shape
    assert (W, H) == (1194, 844), f"ukuran gambar {W}x{H}, skrip dikalibrasi untuk 1194x844"
    dark = a < DARK

    # --- 1. bingkai panel -------------------------------------------------
    top = dark[:TOP_ROWS]
    vcols = [x for x in range(W) if top[:, x].sum() > 200]
    vlines = clusters(vcols, gap=3)
    hrows = [y for y in range(TOP_ROWS) if top[y].sum() > 800]
    hlines = clusters(hrows, gap=3)
    assert len(vlines) == 4, f"diharapkan 4 garis vertikal bingkai, dapat {vlines}"
    L_x0, L_x1, R_x0, R_x1 = vlines
    y_top, y_bot = hlines[0], hlines[-1]

    # --- 2. sentroid label sumbu ------------------------------------------
    lon_band = dark[10:56]
    lon_cols = [x for x in range(W) if lon_band[:, x].any()]
    lon_c = clusters(lon_cols)
    left_lon = [c for c in lon_c if L_x0 + 50 < c < L_x1]
    right_lon = [c for c in lon_c if R_x0 + 50 < c < R_x1]

    def lat_centroids(x0, x1):
        band = dark[55:530, x0:x1]
        rows = [55 + i for i in range(band.shape[0]) if band[i].any()]
        return clusters(rows)

    left_lat = lat_centroids(0, int(L_x0) - 1)
    right_lat = lat_centroids(int(R_x1) + 2, W)

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
        if not (4 <= h <= 12 and 4 <= w <= 12 and abs(h - w) <= 2):
            continue
        fill = (lab[sl] == i).sum() / (h * w)
        cy, cx = (ys.start + ys.stop - 1) / 2, (xs.start + xs.stop - 1) / 2
        inside = y_top < cy < y_bot and (L_x0 < cx < L_x1 or R_x0 < cx < R_x1)
        if fill > 0.80 and inside:
            markers.append((cx, cy))
    assert len(markers) == 8, f"diharapkan 8 marker, dapat {len(markers)}"

    # Penugasan label diverifikasi visual (perbesaran 4x).
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

    print("Validasi:")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Bias bujur dikoreksi: {bias_lon:+.5f} deg")
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
