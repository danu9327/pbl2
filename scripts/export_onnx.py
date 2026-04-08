"""
best.pt → best.onnx 익스포트 (공식 YOLOv8 ONNX 포맷)
=====================================================

출력 포맷:
  입력:  images     [1, 3, 640, 640]  float32  (0~1 정규화)
  출력:  output0    [1, 7, 8400]      float32
           └─ [cx, cy, w, h, bicycle, kickboard, bollard]
              좌표는 픽셀 단위 (0~640), NMS 미포함

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
PT_PATH      = PROJECT_ROOT / "models" / "safewalk_v1" / "weights" / "best.pt"
ONNX_PATH    = PROJECT_ROOT / "models" / "safewalk_v1" / "weights" / "best.onnx"

if not PT_PATH.exists():
    print(f"[ERROR] {PT_PATH} 없음")
    sys.exit(1)

print(f"익스포트: {PT_PATH} → {ONNX_PATH}")
model = YOLO(str(PT_PATH))
model.export(
    format="onnx",
    imgsz=640,       # 공식 YOLOv8 기본값, 학습 크기와 동일
    opset=17,        # 공식 ultralytics 기본 opset
    simplify=True,
    dynamic=False,   # 고정 배치 (batch=1)
    half=False,      # FP32 (공식 포맷)
    int8=False,
    nms=False,       # NMS 미포함 (표준 ONNX 포맷)
)

print()
print("=" * 50)
print(" 완료")
print("=" * 50)
print(f" 파일  : {ONNX_PATH}")
print(f" 입력  : images [1, 3, 640, 640] float32")
print(f" 출력  : output0 [1, 7, 8400] float32")
print(f"         └ [cx, cy, w, h, bicycle, kickboard, bollard]")
print(f" opset : 17")
print("=" * 50)
