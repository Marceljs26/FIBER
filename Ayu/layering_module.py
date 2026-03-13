import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import threshold_multiotsu, gaussian
from skimage import morphology, measure
import argparse, os, json

CONFIG = {
    "n_classes":       4,
    "min_fiber_area":  30,
    "max_hole_area":   800,
    "gaussian_sigma":  1.0,
    "footer_fraction": 0.91,   # fallback: pakai 91% tinggi gambar
}

LAYER_INFO = {
    'layer1_top':  {'color': [220, 50,  50],  'label': 'Layer 1 — Paling Atas'},
    'layer2_mid':  {'color': [255, 165,  0],  'label': 'Layer 2 — Tengah'},
    'layer3_deep': {'color': [ 70, 130, 180], 'label': 'Layer 3 — Dalam'},
    'background':  {'color': [ 40,  40,  40], 'label': 'Background'},
}

# ──────────────────────────────────────────────────────────────
# DETEKSI FOOTER — 3 LAPIS PRIORITAS
# ──────────────────────────────────────────────────────────────

def _footer_from_metadata(meta_path, img_height):
    """
    Lapis 1: Baca DataSize dari file .txt metadata SEM.
    Field DataSize=WxH memberi tahu tinggi area gambar aktual.
    Selisih dengan tinggi file = footer.
    """
    if not meta_path or not os.path.exists(meta_path):
        return None, "file metadata tidak ditemukan"
    with open(meta_path, 'r', errors='ignore') as f:
        for line in f:
            if line.strip().startswith('DataSize='):
                val = line.strip().split('=')[1]      # "1280x960"
                parts = val.lower().split('x')
                if len(parts) == 2:
                    try:
                        data_h = int(parts[1])
                        if data_h < img_height:
                            return data_h, f"DataSize={val} → footer={img_height-data_h} baris"
                        return None, f"DataSize={val} sama dengan tinggi gambar, tidak ada footer"
                    except ValueError:
                        pass
    return None, "DataSize tidak ditemukan di metadata"


def _footer_from_pixels(img):
    """
    Lapis 2: Deteksi footer dari piksel — KONSERVATIF.
    Cari blok gelap seragam (dark_ratio > 0.93) minimal 3 baris
    di 20% bawah gambar. Tidak agresif agar serat tidak terbuang.
    """
    img_f    = img.astype(np.float32)
    img_norm = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-6) * 255.0
    h        = img_norm.shape[0]
    dark_ratio = (img_norm < 255 * 0.18).mean(axis=1)

    scan_from = int(h * 0.80)
    for i in range(scan_from, h):
        if dark_ratio[i] > 0.93:
            block_end = i
            while block_end < h and dark_ratio[block_end] > 0.93:
                block_end += 1
            if (block_end - i) >= 3:
                return i, f"pixel dark_ratio>0.93, baris {i}-{block_end-1}"
    return None, "tidak terdeteksi dari piksel"


def _footer_fallback(img_height):
    """Lapis 3: Fallback proporsi tetap 91% — selalu aman."""
    boundary = int(img_height * CONFIG["footer_fraction"])
    return boundary, f"fallback {CONFIG['footer_fraction']*100:.0f}% tinggi gambar"


def detect_footer_boundary(img, meta_path=None, crop_rows=None):
    """
    Deteksi batas footer SEM — 3 lapis prioritas:

      [1] crop_rows manual  → override langsung jika pengguna tahu pasti
      [2] Metadata .txt     → baca DataSize (paling akurat)
      [3] Deteksi piksel    → blok gelap seragam di bawah (konservatif)
      [4] Fallback 91%      → selalu aman, tidak pernah gagal

    Returns:
      boundary (int) : indeks baris awal footer
      info     (dict): metode yang digunakan dan detail
    """
    h = img.shape[0]

    # [1] Override manual
    if crop_rows is not None:
        b = h - int(crop_rows)
        return b, {'method': f'manual (crop_rows={crop_rows})', 'footer_rows': h - b}

    # [2] Metadata
    b, msg = _footer_from_metadata(meta_path, h)
    if b is not None:
        return b, {'method': f'metadata: {msg}', 'footer_rows': h - b}

    # [3] Piksel
    b, msg = _footer_from_pixels(img)
    if b is not None:
        return b, {'method': f'pixel: {msg}', 'footer_rows': h - b}

    # [4] Fallback
    b, msg = _footer_fallback(h)
    return b, {'method': msg, 'footer_rows': h - b}


# ──────────────────────────────────────────────────────────────
# LOAD & PREPARE
# ──────────────────────────────────────────────────────────────

def load_image(path, meta_path=None, crop_rows=None):
    """
    Load gambar SEM, normalisasi ke uint8, crop footer otomatis.
    Returns: (img_cropped, boundary, footer_info)
    """
    raw = np.array(Image.open(path))
    if raw.ndim == 3:
        raw = raw.mean(axis=2).astype(np.uint8)
    if raw.dtype != np.uint8:
        raw = ((raw.astype(np.float32) - raw.min()) /
               (raw.max() - raw.min() + 1e-6) * 255).astype(np.uint8)

    boundary, footer_info = detect_footer_boundary(raw, meta_path, crop_rows)
    return raw[:boundary, :], boundary, footer_info


# ──────────────────────────────────────────────────────────────
# LAYERING
# ──────────────────────────────────────────────────────────────

def compute_thresholds(img):
    smoothed = (gaussian(img, sigma=CONFIG["gaussian_sigma"]) * 255).astype(np.uint8)
    return threshold_multiotsu(smoothed, classes=CONFIG["n_classes"])


def apply_layering(img, thresholds):
    T1, T2, T3 = thresholds
    raw_masks = {
        'background':  img < T1,
        'layer3_deep': (img >= T1) & (img < T2),
        'layer2_mid':  (img >= T2) & (img < T3),
        'layer1_top':  img >= T3,
    }
    cleaned = {}
    for name, mask in raw_masks.items():
        c = morphology.remove_small_objects(mask, max_size=CONFIG["min_fiber_area"])
        c = morphology.remove_small_holes(c,     max_size=CONFIG["max_hole_area"])
        cleaned[name] = c
    return cleaned


def get_layer_stats(layers, img):
    stats = {}
    for name, mask in layers.items():
        labeled = measure.label(mask)
        props   = measure.regionprops(labeled, intensity_image=img)
        regions = [p for p in props if p.area > CONFIG["min_fiber_area"]]
        stats[name] = {
            'pixel_count':    int(mask.sum()),
            'percentage':     float(round(100 * mask.mean(), 2)),
            'n_regions':      len(regions),
            'mean_intensity': float(round(img[mask].mean(), 2)) if mask.any() else 0.0,
        }
    return stats


# ──────────────────────────────────────────────────────────────
# VISUALISASI
# ──────────────────────────────────────────────────────────────

def make_color_overlay(img, layers):
    rgb = np.stack([img] * 3, axis=-1).copy()
    for name, info in LAYER_INFO.items():
        if name in layers and name != 'background':
            mask = layers[name]
            for c, val in enumerate(info['color']):
                rgb[:, :, c] = np.where(mask, val, rgb[:, :, c])
    return rgb


def visualize(img, layers, thresholds, stats, footer_info, save_path):
    T1, T2, T3 = thresholds
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f'Layering Nanofiber SEM  —  T1={T1:.0f} | T2={T2:.0f} | T3={T3:.0f}\n'
        f'Footer: {footer_info["method"]}  ({footer_info["footer_rows"]} baris dibuang)',
        fontsize=12, fontweight='bold'
    )
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.12)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img, cmap='gray')
    ax.set_title('SEM Image (setelah crop footer)', fontweight='bold')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(img.ravel(), bins=256, color='steelblue', alpha=0.8, density=True)
    for t, c, lb in zip(thresholds, ['green','orange','red'],
                        ['T1 BG|L3','T2 L3|L2','T3 L2|L1']):
        ax.axvline(t, color=c, linestyle='--', linewidth=2, label=f'{lb}={t:.0f}')
    ax.legend(fontsize=9)
    ax.set_title('Histogram Intensitas\n(gelap=dalam, terang=atas)', fontweight='bold')
    ax.set_xlabel('Intensitas'); ax.set_ylabel('Density')

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(make_color_overlay(img, layers))
    ax.set_title('Layer Overlay\nMerah=L1 | Oranye=L2 | Biru=L3', fontweight='bold')
    ax.axis('off')

    for col, (name, cmap_n) in enumerate([
        ('layer1_top', 'Reds'), ('layer2_mid', 'Oranges'), ('layer3_deep', 'Blues')
    ]):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(img, cmap='gray', alpha=0.4)
        vis = np.zeros_like(img); vis[layers[name]] = 255
        ax.imshow(vis, cmap=cmap_n, alpha=0.65)
        s = stats[name]
        ax.set_title(f'{LAYER_INFO[name]["label"]}\n'
                     f'{s["percentage"]:.1f}% area | {s["n_regions"]} region',
                     fontweight='bold', fontsize=10)
        ax.axis('off')

    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Visualisasi: {save_path}")


# ──────────────────────────────────────────────────────────────
# FUNGSI UTAMA
# ──────────────────────────────────────────────────────────────

def run_layering(input_path, output_dir='.', meta_path=None, crop_rows=None):
    """
    Jalankan pipeline layering lengkap.

    Parameters:
      input_path : path gambar SEM (.tif / .png)
      output_dir : folder output hasil
      meta_path  : path metadata .txt SEM (direkomendasikan)
      crop_rows  : override manual baris footer yang dibuang

    Returns:
      layers, thresholds, stats
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    print(f"\n{'='*55}")
    print(f"[1] Load  : {os.path.basename(input_path)}")
    img, boundary, footer_info = load_image(input_path, meta_path, crop_rows)
    print(f"    Shape : {img.shape} | dtype: {img.dtype}")
    print(f"    Footer: {footer_info['method']}")
    print(f"    Dibuang: {footer_info['footer_rows']} baris")

    print(f"[2] Threshold (Multi-Otsu)...")
    thresholds = compute_thresholds(img)
    print(f"    T1={thresholds[0]:.0f}  T2={thresholds[1]:.0f}  T3={thresholds[2]:.0f}")

    print(f"[3] Layering & cleaning...")
    layers = apply_layering(img, thresholds)

    print(f"[4] Statistik:")
    stats = get_layer_stats(layers, img)
    for name, s in stats.items():
        print(f"    {name:15s}: {s['pixel_count']:>8,} px "
              f"({s['percentage']:5.1f}%) | "
              f"{s['n_regions']:2d} region | "
              f"mean={s['mean_intensity']:.1f}")

    print(f"[5] Simpan output...")
    visualize(img, layers, thresholds, stats, footer_info,
              os.path.join(output_dir, f'{base}_layering.png'))

    for name, mask in layers.items():
        Image.fromarray((mask.astype(np.uint8) * 255)).save(
            os.path.join(output_dir, f'{base}_{name}.png'))

    with open(os.path.join(output_dir, f'{base}_stats.json'), 'w') as f:
        json.dump({'config': CONFIG, 'footer_info': footer_info,
                   'thresholds': [float(t) for t in thresholds],
                   'layers': stats}, f, indent=2)
    print(f"{'='*55}\n")

    return layers, thresholds, stats


# ─── CLI ─────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Nanofiber SEM Layering v2.0')
    p.add_argument('--input',     required=True)
    p.add_argument('--output',    default='output_layering')
    p.add_argument('--meta',      default=None, help='Path file .txt metadata SEM')
    p.add_argument('--crop_rows', default=None, type=int,
                   help='Override manual: jumlah baris footer yang dibuang')
    args = p.parse_args()
    run_layering(args.input, args.output, args.meta, args.crop_rows)

