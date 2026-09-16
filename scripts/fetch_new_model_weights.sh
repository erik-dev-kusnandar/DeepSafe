#!/usr/bin/env bash
# Download pretrained weights for the replacement video models (UnivFD + LipForensics + FTCN+TT).
# TALL++ (rainy-xu/TALL4Deepfake) does NOT ship a public pretrained checkpoint,
# so it is skipped here (needs training on FF++ yourself or contacting the authors).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/model_weights_new"
mkdir -p "$OUT"
cd "$OUT"

if ! command -v gdown >/dev/null 2>&1; then
  echo "Installing gdown..."
  python3 -m pip install -q gdown
fi

echo "== 1/2 UnivFD linear head (fc_weights.pth, ~4 KB) =="
curl -fL --retry 3 -o fc_weights.pth \
  https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/main/pretrained_weights/fc_weights.pth
echo "   -> $(du -h fc_weights.pth | cut -f1)"

echo "== 2/2 LipForensics (lipforensics_ff.pth, Google Drive) =="
gdown "https://drive.google.com/uc?id=1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7" -O lipforensics_ff.pth
echo "   -> $(du -h lipforensics_ff.pth | cut -f1)"

echo "== 3/3 FTCN+TT (ftcn_tt.pth, GitHub Release) =="
curl -fL --retry 3 -o ftcn_tt.pth \
  https://github.com/yinglinzheng/FTCN/releases/download/weights/ftcn_tt.pth
echo "   -> $(du -h ftcn_tt.pth | cut -f1)"

echo
echo "Selesai. Berkas tersimpan di: $OUT"
echo "Note: UnivFD backbone (CLIP ViT-L/14, OpenAI) di-download otomatis oleh open_clip pada saat run pertama."
echo "Note: TALL++ tidak punya pretrained publik -> dilewati; FTCN+TT dipakai sebagai penggantinya."
ls -lh "$OUT"