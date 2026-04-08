"""
best.pt → best.onnx 재익스포트
좌표 버그 수정 (imgsz=640, dynamic=False)

사용법:
  python scripts/export_onnx.py
"""

import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] pip install ultralytics")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
PT_PATH   = PROJECT_ROOT / "models" / "safewalk_v1" / "weights" / "best.pt"
ONNX_PATH = PROJECT_ROOT / "models" / "safewalk_v1" / "weights" / "best.onnx"

if not PT_PATH.exists():
    print(f"[ERROR] {PT_PATH} 없음")
    sys.exit(1)

print(f"익스포트: {PT_PATH} → {ONNX_PATH}")
model = YOLO(str(PT_PATH))
model.export(
    format="onnx",
    imgsz=640,        # 학습과 동일한 크기로 맞춤 (기존 320이 좌표 버그 원인)
    simplify=True,
    opset=12,
    dynamic=False,
)
print(f"완료: {ONNX_PATH}")
