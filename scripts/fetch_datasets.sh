#!/usr/bin/env bash
# conger 最小评估包重下载脚本 (2026-08-10)。
# 数据落点 /tmp/datasets (与 demo_bsd_eval 的 /tmp/datasets/BSDS500 同惯例:
# /tmp 可被系统清理, 本脚本幂等可重放)。
#
# 用法:
#   scripts/fetch_datasets.sh            # 补缺 (已有跳过, 校验 iBims sha512)
#   scripts/fetch_datasets.sh --force    # 全部重下
#   scripts/fetch_datasets.sh --skip-bsds
#
# 需要: curl, tar, unzip, gunzip, shasum。凭据: iBims FTP (发布方公开的 m1455541)。

set -euo pipefail

DEST="/tmp/datasets"
FORCE=0
SKIP_BSDS=0
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    --skip-bsds) SKIP_BSDS=1 ;;
    *) echo "未知参数: $a" >&2; exit 2 ;;
  esac
done
mkdir -p "$DEST"

need() { command -v "$1" >/dev/null || { echo "缺少 $1" >&2; exit 1; }; }
for c in curl tar unzip gunzip shasum; do need "$c"; done

fetch() { # fetch <url> <out> [--range <bytes>]
  local url="$1" out="$2"; shift 2
  if [ -f "$out" ] && [ "$FORCE" -eq 0 ]; then
    echo "  已有 $out, 跳过"
    return 0
  fi
  echo "  下载 $url -> $out"
  curl -sL "$@" -o "$out" "$url"
}

# ── 1. BSDS500 (轮廓/区域基准, demo_bsd_eval 读 /tmp/datasets/BSDS500) ──────
fetch_bsds() {
  local url="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_full.tgz"
  local dir="$DEST/BSDS500"
  local want="$dir/BSDS500/data/images/test"
  if [ -d "$want" ] && [ "$(ls "$want"/*.jpg 2>/dev/null | wc -l)" -ge 200 ]; then
    echo "BSDS500 已存在 ($want), 跳过"
    return 0
  fi
  [ "$FORCE" -eq 1 ] && rm -rf "$dir"
  local tgz="/tmp/datasets/BSDS500_src.tgz"
  fetch "$url" "$tgz"
  echo "  解压 -> $dir (BSR/bsds500 -> BSDS500 映射)"
  rm -rf "$dir" && mkdir -p "$dir"
  tar -xzf "$tgz" -C "$dir" --strip-components=1
  # BSR_full.tgz 内部为 BSR/bsds500/data/...; demo 期望 $dir/BSDS500/data/...
  if [ -d "$dir/bsds500" ]; then
    mv "$dir/bsds500" "$dir/BSDS500"
  fi
  rm -f "$tgz"
  echo "  BSDS500 完成: $(ls "$want"/*.jpg 2>/dev/null | wc -l) 测试图"
}

# ── 2. iBims-1 (图元/深度/边界对齐, FTP 凭据为发布方公开值) ────────
fetch_ibims() {
  local base="ftp://dataserv.ub.tum.de"
  local user="m1455541"; local pass="m1455541"
  local dir="$DEST/ibims1"
  mkdir -p "$dir"
  local files="ibims1_core_raw.zip ibims1_core_mat.zip imagelist.txt Ibims_Dataset.sha512 evaluation_scripts.zip"
  for f in $files; do
    if [ -f "$dir/$f" ] && [ "$FORCE" -eq 0 ]; then continue; fi
    echo "  下载 $f"
    curl -s --user "$user:$pass" -o "$dir/$f" "$base/$f"
  done
  if [ -f "$dir/ibims1_core_raw.zip" ]; then
    echo "  校验 sha512:"
    (cd "$dir" && for f in ibims1_core_raw.zip ibims1_core_mat.zip; do
      want=$(grep -E " $f" Ibims_Dataset.sha512 | sed 's/^[0-9]*://; s/\r$//' | awk '{print $1}')
      got=$(shasum -a 512 "$f" | awk '{print $1}')
      if [ "$want" = "$got" ]; then echo "    $f OK"; else echo "    $f 校验失败!" >&2; exit 1; fi
    done)
  fi
  if [ "$FORCE" -eq 1 ] || [ ! -d "$dir/ibims1_core_raw/rgb" ]; then
    echo "  解压"
    (cd "$dir" && unzip -q -o ibims1_core_raw.zip && unzip -q -o ibims1_core_mat.zip \
      && unzip -q -o evaluation_scripts.zip -d eval_scripts)
  fi
  echo "  iBims-1 完成: $(ls "$dir"/ibims1_core_raw/rgb/*.png 2>/dev/null | wc -l) 场景"
}

# ── 3. NYUv2 Eigen 654 测试切分 (HF 镜像, rgb+depth 米制 h5) ───────
fetch_nyu() {
  local base="https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/main/data"
  local dir="$DEST/nyu_v2_eigen"
  mkdir -p "$dir"
  for f in val-000000.tar val-000001.tar; do
    fetch "$base/$f" "$dir/$f"
  done
  local out="$dir/extract/val/official"
  if [ "$FORCE" -eq 1 ] || [ "$(ls "$out"/*.h5 2>/dev/null | wc -l)" -lt 600 ]; then
    echo "  解压"
    rm -rf "$dir/extract" && mkdir -p "$dir/extract"
    (cd "$dir" && tar -xf val-000000.tar -C extract && tar -xf val-000001.tar -C extract)
  fi
  echo "  NYUv2 完成: $(ls "$out"/*.h5 2>/dev/null | wc -l) h5 (期望 654)"
}

# ── 4. OASIS 野图前缀子集 (Range 250MB; 标注需全量 31GB, 见 README) ─
fetch_oasis() {
  local url="https://pvl-weifengc.cs.princeton.edu/OASIS/OASIS_images_v1.tar.gz"
  local dir="$DEST/oasis"
  local raw="$dir/raw"; mkdir -p "$raw" "$dir/images"
  local n=$(ls "$dir"/images/*.png 2>/dev/null | wc -l)
  if [ "$n" -ge 50 ] && [ "$FORCE" -eq 0 ]; then
    echo "OASIS 已存在 $n 张, 跳过"
    return 0
  fi
  rm -rf "$raw" "$dir/images" && mkdir -p "$raw" "$dir/images"
  fetch "$url" "$raw/img_prefix.gz" --range 0-262143999
  echo "  解压前缀 (gzip 截断属预期)"
  gunzip -c "$raw/img_prefix.gz" > "$raw/img_prefix.tar" 2>/dev/null || true
  (cd "$raw" && tar -xf img_prefix.tar 2>/dev/null || true)
  local src="$raw/OASIS_trainval/image"
  local k=0
  for f in "$src"/*.png; do
    case "$(basename "$f")" in *_DT.png) continue;; esac
    cp "$f" "$dir/images/" && k=$((k+1))
  done
  rm -rf "$raw"
  echo "OASIS 完成: $k 张 (期望 ≥50; 无 GT, 仅定性)"
}

echo "== conger 评估包重下载 (目标 $DEST, force=$FORCE) =="
fetch_bsds
fetch_ibims
fetch_nyu
fetch_oasis
echo "== 完成。清单见 $DEST/README.md =="
