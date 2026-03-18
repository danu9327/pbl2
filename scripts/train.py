"""
SafeWalk Navigation - Phase 2: YOLO Training Pipeline
=====================================================

YOLOv8n 학습 → 평가 → TFLite 변환 (모바일 배포용)

사용법:
    # Step 1: 학습
    python scripts/train.py train

    # Step 2: 평가
    python scripts/train.py evaluate

    # Step 3: TFLite 변환 (모바일용)
    python scripts/train.py export

    # 전체 한번에
    python scripts/train.py all
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] pip install ultralytics")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ============================================
# 학습 설정
# ============================================
TRAIN_CONFIG = {
    # 모델
    "model": "yolov8n.pt",           # nano (가장 경량)

    # 데이터
    "data": str(CONFIG_PATH),

    # 학습 하이퍼파라미터
    "epochs": 100,                    # 학습 에포크 (데이터 규모 대비 충분)
    "batch": 16,                      # 배치 사이즈 (GPU 메모리에 맞춰 조정)
    "imgsz": 640,                     # 학습 이미지 크기 (정확도 위해 640 유지)
    "patience": 20,                   # Early stopping (20 에포크 개선 없으면 중단)

    # 옵티마이저
    "optimizer": "AdamW",
    "lr0": 0.001,                     # 초기 learning rate
    "lrf": 0.01,                      # 최종 lr = lr0 * lrf
    "weight_decay": 0.0005,
    "warmup_epochs": 5,

    # 증강 (YOLO 내장)
    "mosaic": 1.0,                    # Mosaic 증강
    "mixup": 0.1,                     # MixUp 증강
    "degrees": 10.0,                  # 회전
    "translate": 0.2,                 # 이동
    "scale": 0.5,                     # 스케일
    "flipud": 0.0,                    # 상하 반전 (보행 환경에선 비현실적)
    "fliplr": 0.5,                    # 좌우 반전
    "hsv_h": 0.015,                   # 색상 변화
    "hsv_s": 0.5,                     # 채도 변화
    "hsv_v": 0.4,                     # 명도 변화

    # 저장
    "project": str(MODELS_DIR),
    "name": "safewalk_v1",
    "exist_ok": True,
    "save": True,
    "save_period": 10,                # 10 에포크마다 체크포인트
    "plots": True,                    # 학습 그래프 저장
    "verbose": True,
}


# ============================================
# Step 1: 학습
# ============================================
def train(resume: bool = False):
    """YOLOv8n 학습"""
    print(f"\n{'='*55}")
    print(f" SafeWalk YOLO Training")
    print(f"{'='*55}")
    print(f" Model:    YOLOv8n (nano)")
    print(f" Dataset:  {CONFIG_PATH}")
    print(f" Epochs:   {TRAIN_CONFIG['epochs']}")
    print(f" ImgSize:  {TRAIN_CONFIG['imgsz']}")
    print(f" Batch:    {TRAIN_CONFIG['batch']}")
    print(f"{'='*55}\n")

    if resume:
        # 마지막 체크포인트에서 이어서 학습
        last_ckpt = MODELS_DIR / "safewalk_v1" / "weights" / "last.pt"
        if last_ckpt.exists():
            print(f"[INFO] Resuming from {last_ckpt}")
            model = YOLO(str(last_ckpt))
            model.train(resume=True)
        else:
            print("[ERROR] No checkpoint found. Starting fresh.")
            model = YOLO(TRAIN_CONFIG["model"])
            model.train(**TRAIN_CONFIG)
    else:
        model = YOLO(TRAIN_CONFIG["model"])
        model.train(**TRAIN_CONFIG)

    # 결과 경로 안내
    results_dir = MODELS_DIR / "safewalk_v1"
    print(f"\n{'='*55}")
    print(f" Training complete!")
    print(f"{'='*55}")
    print(f" Best weights: {results_dir}/weights/best.pt")
    print(f" Last weights: {results_dir}/weights/last.pt")
    print(f" Metrics:      {results_dir}/results.csv")
    print(f" Plots:        {results_dir}/")
    print(f"{'='*55}")


# ============================================
# Step 2: 평가
# ============================================
def evaluate():
    """학습된 모델 평가"""
    best_pt = MODELS_DIR / "safewalk_v1" / "weights" / "best.pt"

    if not best_pt.exists():
        print(f"[ERROR] Model not found: {best_pt}")
        print(f"  → 먼저 학습을 실행하세요: python scripts/train.py train")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f" Evaluating: {best_pt}")
    print(f"{'='*55}\n")

    model = YOLO(str(best_pt))

    # Validation set 평가
    results = model.val(
        data=str(CONFIG_PATH),
        imgsz=640,
        batch=16,
        split="test",            # test set으로 최종 평가
        plots=True,
        save_json=True,
        project=str(MODELS_DIR),
        name="safewalk_v1_eval",
        exist_ok=True,
    )

    # 결과 요약
    print(f"\n{'='*55}")
    print(f" Evaluation Results")
    print(f"{'='*55}")
    print(f" mAP@0.5:      {results.box.map50:.4f}")
    print(f" mAP@0.5:0.95: {results.box.map:.4f}")

    class_names = ["bicycle", "kickboard", "bollard"]
    if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
        print(f"\n Per-class AP@0.5:")
        for i, name in enumerate(class_names):
            if i < len(results.box.ap50):
                print(f"   {name:<12} {results.box.ap50[i]:.4f}")

    print(f"{'='*55}")


# ============================================
# Step 3: 모바일 변환 (TFLite)
# ============================================
def export_mobile():
    """TFLite INT8 변환 (모바일 온디바이스 추론용)"""
    best_pt = MODELS_DIR / "safewalk_v1" / "weights" / "best.pt"

    if not best_pt.exists():
        print(f"[ERROR] Model not found: {best_pt}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f" Exporting for mobile deployment")
    print(f"{'='*55}\n")

    model = YOLO(str(best_pt))

    # --- ONNX Export ---
    print("[1/3] Exporting ONNX...")
    model.export(
        format="onnx",
        imgsz=320,              # 모바일용 경량 입력 크기
        simplify=True,
        opset=13,
    )
    onnx_path = best_pt.with_suffix('.onnx')
    print(f"  → {onnx_path}")

    # --- TFLite Export (FP16) ---
    print("\n[2/3] Exporting TFLite FP16...")
    model.export(
        format="tflite",
        imgsz=320,
        half=True,              # FP16 양자화
    )

    # --- TFLite Export (INT8) ---
    print("\n[3/3] Exporting TFLite INT8...")
    try:
        model.export(
            format="tflite",
            imgsz=320,
            int8=True,          # INT8 양자화 (가장 경량)
        )
    except Exception as e:
        print(f"  [WARN] INT8 export failed: {e}")
        print(f"  → FP16 모델을 사용하세요")

    # 결과 파일 정리
    export_dir = MODELS_DIR / "safewalk_v1" / "weights"
    print(f"\n{'='*55}")
    print(f" Export complete!")
    print(f"{'='*55}")

    for f in export_dir.iterdir():
        if f.suffix in ['.onnx', '.tflite']:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f" {f.name:<35} {size_mb:.1f} MB")

    print(f"\n 모바일 배포:")
    print(f"   → TFLite INT8 파일을 Flutter assets에 복사")
    print(f"   → 입력 크기: 320x320")
    print(f"   → 클래스: bicycle(0), kickboard(1), bollard(2)")
    print(f"{'='*55}")


# ============================================
# CLI
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description="SafeWalk YOLO Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py train             # 학습 시작
  python scripts/train.py train --resume    # 이어서 학습
  python scripts/train.py evaluate          # 평가
  python scripts/train.py export            # TFLite 변환
  python scripts/train.py all               # 전체 실행
        """
    )
    subparsers = parser.add_subparsers(dest="command")

    # train
    tr = subparsers.add_parser("train", help="YOLOv8n 학습")
    tr.add_argument("--resume", action="store_true", help="이전 학습 이어서")
    tr.add_argument("--epochs", type=int, help="에포크 수 오버라이드")
    tr.add_argument("--batch", type=int, help="배치 사이즈 오버라이드")
    tr.add_argument("--imgsz", type=int, help="이미지 크기 오버라이드")

    # evaluate
    subparsers.add_parser("evaluate", help="모델 평가")

    # export
    subparsers.add_parser("export", help="TFLite 모바일 변환")

    # all
    al = subparsers.add_parser("all", help="train → evaluate → export")
    al.add_argument("--epochs", type=int, help="에포크 수 오버라이드")
    al.add_argument("--batch", type=int, help="배치 사이즈 오버라이드")

    args = parser.parse_args()

    if args.command == "train":
        if args.epochs:
            TRAIN_CONFIG["epochs"] = args.epochs
        if args.batch:
            TRAIN_CONFIG["batch"] = args.batch
        if args.imgsz:
            TRAIN_CONFIG["imgsz"] = args.imgsz
        train(resume=args.resume)

    elif args.command == "evaluate":
        evaluate()

    elif args.command == "export":
        export_mobile()

    elif args.command == "all":
        if args.epochs:
            TRAIN_CONFIG["epochs"] = args.epochs
        if args.batch:
            TRAIN_CONFIG["batch"] = args.batch

        print("\n[1/3] Training...")
        train()

        print("\n[2/3] Evaluating...")
        evaluate()

        print("\n[3/3] Exporting for mobile...")
        export_mobile()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
