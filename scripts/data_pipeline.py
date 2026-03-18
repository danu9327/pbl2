"""
SafeWalk Navigation - Data Pipeline
====================================
Roboflow 3개 데이터셋 다운로드 → 클래스 리매핑 → 통합 → 분할 → 검증

사용법:
    # Step 1: Roboflow 데이터셋 다운로드
    python scripts/data_pipeline.py download --api-key YOUR_ROBOFLOW_API_KEY

    # Step 2: 클래스 리매핑 + 통합
    python scripts/data_pipeline.py merge

    # Step 3: Train/Val/Test 분할
    python scripts/data_pipeline.py split

    # Step 4: 검증 + 통계
    python scripts/data_pipeline.py validate

    # 전체 한번에 실행 (다운로드 제외)
    python scripts/data_pipeline.py all
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


# ============================================
# 프로젝트 경로 설정
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MERGED_DIR = DATA_DIR / "merged"
PROCESSED_DIR = DATA_DIR / "processed"

# ============================================
# 데이터셋 정보 및 클래스 매핑
# ============================================
DATASETS = {
    "bicycle": {
        "workspace": "dng-cjryd",
        "project": "bicycle-i7nhz",
        "version": 1,
        "description": "자전거 (Dng, 127 images)",
        # 원본 class_id → 통합 class_id
        "class_remap": {
            0: 0,  # Bicycle → bicycle
        }
    },
    "kickboard": {
        "workspace": "inha-univ-vgzgz",
        "project": "kickboard-ibhkj",
        "version": 2,
        "description": "킥보드 (Inha Univ, 462 images)",
        "class_remap": {
            0: 1,  # kb → kickboard
        }
    },
    "bollard": {
        "workspace": "project-60htx",
        "project": "bollard-v2gn5",
        "version": 1,
        "description": "볼라드 (project-60htx, 634 images)",
        "class_remap": {
            0: 2,  # bollad1 → bollard
            1: 2,  # bollard → bollard
            2: 2,  # bollard_abnormal → bollard
            3: 2,  # bollard_normal → bollard
            4: 2,  # tubular marker_normal → bollard
        }
    },
}

UNIFIED_CLASSES = {0: "bicycle", 1: "kickboard", 2: "bollard"}


# ============================================
# Step 1: Roboflow 다운로드
# ============================================
def download_datasets(api_key: str):
    """Roboflow에서 3개 데이터셋을 YOLOv8 포맷으로 다운로드"""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)

    for name, info in DATASETS.items():
        print(f"\n{'='*50}")
        print(f"Downloading: {info['description']}")
        print(f"{'='*50}")

        save_path = RAW_DIR / name
        if save_path.exists() and any(save_path.iterdir()):
            print(f"  [SKIP] Already exists: {save_path}")
            continue

        try:
            project = rf.workspace(info["workspace"]).project(info["project"])
            dataset = project.version(info["version"]).download(
                "yolov8",
                location=str(save_path)
            )
            print(f"  [OK] Downloaded to {save_path}")
        except Exception as e:
            print(f"  [ERROR] Failed: {e}")
            print(f"  수동 다운로드: https://universe.roboflow.com/{info['workspace']}/{info['project']}")
            print(f"  → YOLOv8 포맷으로 다운로드 후 {save_path} 에 저장")

    print(f"\n[DONE] 데이터셋 다운로드 완료")


# ============================================
# Step 2: 클래스 리매핑 + 통합
# ============================================
def remap_label_file(
    src_path: Path,
    dst_path: Path,
    class_remap: Dict[int, int]
) -> int:
    """
    단일 라벨 파일의 클래스 ID를 리매핑

    Returns: 리매핑된 bbox 수
    """
    new_lines = []
    with open(src_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            orig_class = int(float(parts[0]))
            if orig_class in class_remap:
                parts[0] = str(class_remap[orig_class])
                new_lines.append(' '.join(parts))

    if new_lines:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, 'w') as f:
            f.write('\n'.join(new_lines) + '\n')

    return len(new_lines)


def merge_datasets():
    """3개 데이터셋을 클래스 리매핑하여 하나로 통합"""
    # 기존 merged 디렉토리 정리
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    img_out = MERGED_DIR / "images"
    lbl_out = MERGED_DIR / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    stats = defaultdict(lambda: {"images": 0, "bboxes": 0, "class_counts": defaultdict(int)})
    total_images = 0
    total_bboxes = 0

    for dataset_name, info in DATASETS.items():
        print(f"\n--- Processing: {dataset_name} ---")
        raw_path = RAW_DIR / dataset_name

        # Roboflow YOLOv8 구조: train/images, train/labels, valid/..., test/...
        for split in ["train", "valid", "test"]:
            split_img_dir = raw_path / split / "images"
            split_lbl_dir = raw_path / split / "labels"

            if not split_img_dir.exists():
                continue

            for img_path in split_img_dir.iterdir():
                if img_path.suffix.lower() not in image_exts:
                    continue

                lbl_path = split_lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    continue

                # 파일명 충돌 방지: dataset_name prefix 추가
                new_name = f"{dataset_name}_{img_path.name}"
                new_lbl_name = f"{dataset_name}_{img_path.stem}.txt"

                # 이미지 복사
                shutil.copy2(img_path, img_out / new_name)

                # 라벨 리매핑 + 복사
                bbox_count = remap_label_file(
                    lbl_path,
                    lbl_out / new_lbl_name,
                    info["class_remap"]
                )

                if bbox_count > 0:
                    total_images += 1
                    total_bboxes += bbox_count
                    stats[dataset_name]["images"] += 1
                    stats[dataset_name]["bboxes"] += bbox_count

                    # 클래스별 카운트
                    with open(lbl_out / new_lbl_name, 'r') as f:
                        for line in f:
                            cls_id = int(line.strip().split()[0])
                            stats[dataset_name]["class_counts"][cls_id] += 1

    # 통계 출력
    print(f"\n{'='*55}")
    print(f" Merge Summary")
    print(f"{'='*55}")
    print(f" {'Dataset':<15} {'Images':>8} {'BBoxes':>8}")
    print(f" {'-'*15} {'-'*8} {'-'*8}")
    for name, s in stats.items():
        print(f" {name:<15} {s['images']:>8} {s['bboxes']:>8}")
    print(f" {'-'*15} {'-'*8} {'-'*8}")
    print(f" {'TOTAL':<15} {total_images:>8} {total_bboxes:>8}")

    print(f"\n Class distribution:")
    all_counts = defaultdict(int)
    for s in stats.values():
        for cls_id, cnt in s["class_counts"].items():
            all_counts[cls_id] += cnt
    for cls_id in sorted(all_counts.keys()):
        name = UNIFIED_CLASSES.get(cls_id, f"unknown_{cls_id}")
        print(f"   {cls_id}: {name:<15} {all_counts[cls_id]:>6} bboxes")

    print(f"\n[OK] Merged → {MERGED_DIR}")
    return total_images


# ============================================
# Step 3: Train/Val/Test Split
# ============================================
def split_dataset(
    ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42
):
    """통합 데이터셋을 train/val/test로 분할 (stratified)"""
    random.seed(seed)

    img_dir = MERGED_DIR / "images"
    lbl_dir = MERGED_DIR / "labels"
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    # 이미지-라벨 쌍 수집 (데이터셋 출처별로 그룹핑)
    groups = defaultdict(list)  # dataset_name → [(img, lbl), ...]
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in image_exts:
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        # prefix로 데이터셋 출처 파악
        for ds_name in DATASETS:
            if img_path.name.startswith(f"{ds_name}_"):
                groups[ds_name].append((img_path, lbl_path))
                break

    # 각 데이터셋별로 비율 맞춰 분할 (Stratified split)
    splits = {"train": [], "val": [], "test": []}

    for ds_name, pairs in groups.items():
        random.shuffle(pairs)
        n = len(pairs)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        splits["train"].extend(pairs[:train_end])
        splits["val"].extend(pairs[train_end:val_end])
        splits["test"].extend(pairs[val_end:])

    # 각 split 내에서도 셔플
    for split_pairs in splits.values():
        random.shuffle(split_pairs)

    # 파일 복사
    for split_name, split_pairs in splits.items():
        split_img = PROCESSED_DIR / "images" / split_name
        split_lbl = PROCESSED_DIR / "labels" / split_name
        split_img.mkdir(parents=True, exist_ok=True)
        split_lbl.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in split_pairs:
            shutil.copy2(img_path, split_img / img_path.name)
            shutil.copy2(lbl_path, split_lbl / lbl_path.name)

    # 통계 출력
    print(f"\n{'='*55}")
    print(f" Split Summary (seed={seed})")
    print(f"{'='*55}")
    print(f" {'Split':<8} {'Images':>8}  Breakdown by source")
    print(f" {'-'*8} {'-'*8}  {'-'*30}")

    for split_name, split_pairs in splits.items():
        source_counts = defaultdict(int)
        for img_path, _ in split_pairs:
            for ds_name in DATASETS:
                if img_path.name.startswith(f"{ds_name}_"):
                    source_counts[ds_name] += 1
                    break

        breakdown = ", ".join(f"{k}:{v}" for k, v in sorted(source_counts.items()))
        print(f" {split_name:<8} {len(split_pairs):>8}  {breakdown}")

    total = sum(len(p) for p in splits.values())
    print(f" {'-'*8} {'-'*8}")
    print(f" {'TOTAL':<8} {total:>8}")
    print(f"\n[OK] Split → {PROCESSED_DIR}")


# ============================================
# Step 4: 데이터셋 검증
# ============================================
def validate_dataset():
    """데이터셋 무결성 + 통계 검증"""
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    issues = []
    total_stats = {}

    for split in ["train", "val", "test"]:
        img_dir = PROCESSED_DIR / "images" / split
        lbl_dir = PROCESSED_DIR / "labels" / split

        if not img_dir.exists():
            issues.append(f"[MISSING] {img_dir}")
            continue

        images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in image_exts}
        labels = {p.stem: p for p in lbl_dir.iterdir() if p.suffix == '.txt'}

        # 쌍 검증
        no_label = set(images.keys()) - set(labels.keys())
        no_image = set(labels.keys()) - set(images.keys())
        if no_label:
            issues.append(f"[{split}] {len(no_label)} images without labels")
        if no_image:
            issues.append(f"[{split}] {len(no_image)} labels without images")

        # 라벨 내용 검증 + 통계
        class_counts = defaultdict(int)
        bbox_total = 0
        bbox_per_image = []

        for stem in sorted(images.keys() & labels.keys()):
            lbl_path = labels[stem]
            with open(lbl_path, 'r') as f:
                lines = f.readlines()

            bbox_count = 0
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    issues.append(f"[{split}] {lbl_path.name}:{i+1} - {len(parts)} values (expected 5)")
                    continue

                try:
                    cls_id = int(float(parts[0]))
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    issues.append(f"[{split}] {lbl_path.name}:{i+1} - parse error")
                    continue

                if cls_id not in UNIFIED_CLASSES:
                    issues.append(f"[{split}] {lbl_path.name}:{i+1} - unknown class {cls_id}")

                for j, val in enumerate(coords):
                    if val < 0 or val > 1.01:  # 약간의 여유
                        coord_name = ['cx', 'cy', 'w', 'h'][j]
                        issues.append(f"[{split}] {lbl_path.name}:{i+1} - {coord_name}={val:.4f}")

                class_counts[cls_id] += 1
                bbox_count += 1

            bbox_total += bbox_count
            bbox_per_image.append(bbox_count)

        avg_bbox = sum(bbox_per_image) / max(len(bbox_per_image), 1)
        total_stats[split] = {
            "images": len(images),
            "bboxes": bbox_total,
            "avg_bbox": avg_bbox,
            "class_counts": dict(class_counts),
        }

    # 결과 출력
    print(f"\n{'='*60}")
    print(f" Validation Report")
    print(f"{'='*60}")

    if issues:
        print(f"\n [ISSUES] {len(issues)} problems found:")
        for issue in issues[:15]:
            print(f"   {issue}")
        if len(issues) > 15:
            print(f"   ... +{len(issues)-15} more")
    else:
        print(f"\n [OK] No issues found!")

    print(f"\n Statistics:")
    print(f" {'Split':<8} {'Images':>8} {'BBoxes':>8} {'Avg/Img':>8}")
    print(f" {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for split, s in total_stats.items():
        print(f" {split:<8} {s['images']:>8} {s['bboxes']:>8} {s['avg_bbox']:>8.1f}")

    print(f"\n Class distribution (all splits):")
    all_counts = defaultdict(int)
    for s in total_stats.values():
        for cls_id, cnt in s["class_counts"].items():
            all_counts[cls_id] += cnt
    total_bbox = sum(all_counts.values())
    for cls_id in sorted(all_counts.keys()):
        name = UNIFIED_CLASSES.get(cls_id, f"unknown_{cls_id}")
        pct = all_counts[cls_id] / max(total_bbox, 1) * 100
        bar = "#" * int(pct / 2)
        print(f"   {cls_id}: {name:<12} {all_counts[cls_id]:>6} ({pct:5.1f}%) {bar}")

    # 클래스 불균형 경고
    counts = list(all_counts.values())
    if counts:
        ratio = max(counts) / max(min(counts), 1)
        if ratio > 5:
            print(f"\n [WARN] Class imbalance ratio: {ratio:.1f}x")
            print(f"   → data augmentation 또는 추가 수집 권장")
            print(f"   → YOLO 학습 시 class weights 조정 고려")

    return len(issues) == 0


# ============================================
# CLI
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description="SafeWalk Navigation - Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/data_pipeline.py download --api-key YOUR_KEY
  python scripts/data_pipeline.py merge
  python scripts/data_pipeline.py split
  python scripts/data_pipeline.py validate
  python scripts/data_pipeline.py all          # merge + split + validate
        """
    )
    subparsers = parser.add_subparsers(dest="command")

    # download
    dl = subparsers.add_parser("download", help="Roboflow 데이터셋 다운로드")
    dl.add_argument("--api-key", required=True, help="Roboflow API key")

    # merge
    subparsers.add_parser("merge", help="클래스 리매핑 + 통합")

    # split
    sp = subparsers.add_parser("split", help="Train/Val/Test 분할")
    sp.add_argument("--ratios", default="0.7,0.2,0.1", help="분할 비율")
    sp.add_argument("--seed", type=int, default=42)

    # validate
    subparsers.add_parser("validate", help="데이터셋 검증 + 통계")

    # all
    al = subparsers.add_parser("all", help="merge + split + validate 한번에")
    al.add_argument("--ratios", default="0.7,0.2,0.1")
    al.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "download":
        download_datasets(args.api_key)

    elif args.command == "merge":
        merge_datasets()

    elif args.command == "split":
        ratios = tuple(float(x) for x in args.ratios.split(","))
        split_dataset(ratios=ratios, seed=args.seed)

    elif args.command == "validate":
        validate_dataset()

    elif args.command == "all":
        print("\n[1/3] Merging datasets...")
        count = merge_datasets()
        if count == 0:
            print("[ERROR] No data merged. Run 'download' first.")
            sys.exit(1)

        print("\n[2/3] Splitting dataset...")
        ratios = tuple(float(x) for x in args.ratios.split(","))
        split_dataset(ratios=ratios, seed=args.seed)

        print("\n[3/3] Validating dataset...")
        validate_dataset()

        print(f"\n{'='*55}")
        print(f" Pipeline complete!")
        print(f" Dataset ready at: {PROCESSED_DIR}")
        print(f" Config file: configs/dataset.yaml")
        print(f"{'='*55}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
