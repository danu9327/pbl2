"""
SafeWalk Navigation - Data Augmentation
========================================
클래스 불균형 해소를 위한 타겟 증강

bicycle(127장)이 kickboard(462장), bollard(634장)보다 훨씬 적으므로
부족한 클래스를 집중적으로 증강합니다.

사용법:
    # 부족한 클래스 자동 감지 + 증강
    python scripts/augmentation.py --target-per-class 500

    # 특정 클래스만 증강
    python scripts/augmentation.py --classes 0 --multiply 4

설치:
    pip install albumentations opencv-python-headless
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

try:
    import albumentations as A
except ImportError:
    print("pip install albumentations opencv-python-headless")
    exit(1)


PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
UNIFIED_CLASSES = {0: "bicycle", 1: "kickboard", 2: "bollard"}


def get_transform():
    """보행 환경 특화 증강 파이프라인"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
        ], p=0.7),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
        ], p=0.25),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1),
        ], p=0.2),
        A.Affine(rotate=(-8, 8), shear=(-5, 5), scale=(0.9, 1.1), p=0.3),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 2), p=0.2),
        A.ImageCompression(quality_lower=65, quality_upper=95, p=0.15),
    ],
    bbox_params=A.BboxParams(
        format='yolo', label_fields=['class_labels'],
        min_area=100, min_visibility=0.3
    ))


def count_classes(label_dir: Path) -> dict:
    """라벨 디렉토리의 클래스별 이미지 수 카운트"""
    class_images = defaultdict(set)  # cls_id → set of image stems
    for lbl_file in label_dir.glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    class_images[cls_id].add(lbl_file.stem)
    return {k: len(v) for k, v in class_images.items()}


def get_images_by_class(label_dir: Path) -> dict:
    """클래스별 이미지 파일 stem 목록"""
    class_stems = defaultdict(list)
    for lbl_file in label_dir.glob("*.txt"):
        with open(lbl_file) as f:
            classes_in_file = set()
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    classes_in_file.add(int(float(parts[0])))
        for cls_id in classes_in_file:
            class_stems[cls_id].append(lbl_file.stem)
    return class_stems


def augment_images(
    image_dir: Path,
    label_dir: Path,
    stems: list,
    multiply: int,
    prefix: str = "aug"
) -> int:
    """선택된 이미지들을 증강"""
    transform = get_transform()
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    generated = 0

    for stem in stems:
        # 이미지 파일 찾기
        img_path = None
        for ext in image_exts:
            candidate = image_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue

        lbl_path = label_dir / f"{stem}.txt"
        if not lbl_path.exists():
            continue

        # 이미지 + 라벨 읽기
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        class_labels = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_labels.append(int(float(parts[0])))
                    bboxes.append([float(x) for x in parts[1:]])

        # 증강 생성
        for i in range(multiply):
            try:
                result = transform(
                    image=image,
                    bboxes=bboxes if bboxes else [],
                    class_labels=class_labels if class_labels else []
                )

                aug_name = f"{prefix}_{stem}_{i}"
                aug_img_path = image_dir / f"{aug_name}{img_path.suffix}"
                aug_lbl_path = label_dir / f"{aug_name}.txt"

                cv2.imwrite(
                    str(aug_img_path),
                    cv2.cvtColor(result['image'], cv2.COLOR_RGB2BGR)
                )

                if result['bboxes']:
                    with open(aug_lbl_path, 'w') as f:
                        for cls_id, bbox in zip(result['class_labels'], result['bboxes']):
                            f.write(f"{int(cls_id)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                generated += 1
            except Exception:
                continue

    return generated


def auto_balance(target_per_class: int = 500, seed: int = 42):
    """클래스 불균형 자동 해소"""
    np.random.seed(seed)

    train_img = PROCESSED_DIR / "images" / "train"
    train_lbl = PROCESSED_DIR / "labels" / "train"

    if not train_img.exists():
        print("[ERROR] Train data not found. Run 'split' first.")
        return

    # 현재 클래스 분포
    counts = count_classes(train_lbl)
    class_stems = get_images_by_class(train_lbl)

    print(f"\n Current class distribution (train):")
    for cls_id in sorted(UNIFIED_CLASSES.keys()):
        name = UNIFIED_CLASSES[cls_id]
        cnt = counts.get(cls_id, 0)
        print(f"   {cls_id}: {name:<12} {cnt:>5} images")

    print(f"\n Target: {target_per_class} images per class")

    total_generated = 0
    for cls_id in sorted(UNIFIED_CLASSES.keys()):
        name = UNIFIED_CLASSES[cls_id]
        current = counts.get(cls_id, 0)

        if current >= target_per_class:
            print(f"\n   {name}: already {current} >= {target_per_class}, skip")
            continue

        need = target_per_class - current
        stems = class_stems.get(cls_id, [])
        if not stems:
            print(f"\n   {name}: no source images, skip")
            continue

        # 필요한 증강 배수 계산
        multiply = max(1, (need + len(stems) - 1) // len(stems))
        print(f"\n   {name}: {current} → {target_per_class} (need +{need}, {multiply}x aug)")

        generated = augment_images(
            train_img, train_lbl, stems, multiply, prefix=f"bal{cls_id}"
        )
        total_generated += generated
        print(f"   → Generated {generated} augmented images")

    # 최종 통계
    new_counts = count_classes(train_lbl)
    print(f"\n Final distribution:")
    for cls_id in sorted(UNIFIED_CLASSES.keys()):
        name = UNIFIED_CLASSES[cls_id]
        old = counts.get(cls_id, 0)
        new = new_counts.get(cls_id, 0)
        diff = new - old
        print(f"   {cls_id}: {name:<12} {old:>5} → {new:>5} (+{diff})")

    print(f"\n[OK] Total augmented: {total_generated} images")


def main():
    parser = argparse.ArgumentParser(description="SafeWalk Data Augmentation")
    parser.add_argument("--target-per-class", type=int, default=500,
                        help="클래스별 목표 이미지 수 (default: 500)")
    parser.add_argument("--classes", type=int, nargs="+",
                        help="증강할 특정 클래스 ID (미지정 시 자동 밸런싱)")
    parser.add_argument("--multiply", type=int, default=3,
                        help="증강 배수 (--classes와 함께 사용)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.classes:
        # 특정 클래스만 증강
        train_img = PROCESSED_DIR / "images" / "train"
        train_lbl = PROCESSED_DIR / "labels" / "train"
        class_stems = get_images_by_class(train_lbl)

        for cls_id in args.classes:
            name = UNIFIED_CLASSES.get(cls_id, f"class_{cls_id}")
            stems = class_stems.get(cls_id, [])
            print(f"\nAugmenting {name} ({len(stems)} images, {args.multiply}x)...")
            gen = augment_images(train_img, train_lbl, stems, args.multiply)
            print(f"  Generated: {gen}")
    else:
        auto_balance(target_per_class=args.target_per_class, seed=args.seed)


if __name__ == "__main__":
    main()
