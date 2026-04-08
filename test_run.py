"""
ONNX 모델 객체탐지 테스트
사용법:
  python test_run.py                        # 웹캠 1프레임
  python test_run.py --source image.jpg     # 이미지 파일
"""

import argparse
import sys
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] pip install ultralytics")
    sys.exit(1)

MODEL_PATH = Path(__file__).parent / "models" / "safewalk_v1" / "weights" / "best.onnx"
CLASS_NAMES = {0: "bicycle", 1: "kickboard", 2: "bollard"}
COLORS = {0: (0, 200, 255), 1: (0, 255, 100), 2: (255, 100, 0)}


def run(source: str, conf: float):
    if not MODEL_PATH.exists():
        print(f"[ERROR] 모델 없음: {MODEL_PATH}")
        sys.exit(1)

    print(f"모델: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH), task="detect")

    # 이미지 로드
    if source == "0":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] 웹캠을 열 수 없습니다. --source 로 이미지 경로를 지정하세요.")
            sys.exit(1)
        # 웹캠 초기화 대기 (첫 몇 프레임은 검은 화면)
        for _ in range(10):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("[ERROR] 웹캠 프레임 캡처 실패")
            sys.exit(1)
        print("소스: 웹캠 (1프레임 캡처)")
    else:
        frame = cv2.imread(source)
        if frame is None:
            print(f"[ERROR] 이미지를 열 수 없음: {source}")
            sys.exit(1)
        print(f"소스: {source}")

    # 추론
    h, w = frame.shape[:2]
    print(f"이미지 크기: {w}x{h}")

    results = model(frame, conf=conf, verbose=False)[0]
    boxes = results.boxes

    print(f"\n{'='*40}")
    print(f" 탐지 결과 (conf >= {conf})")
    print(f"{'='*40}")

    if len(boxes) == 0:
        print(" 탐지된 객체 없음")
    else:
        for i, box in enumerate(boxes):
            cls  = int(box.cls[0])
            cf   = float(box.conf[0])
            raw  = box.xyxy[0].tolist()
            print(f" [{i+1}] raw xyxy: {raw}")   # 좌표 원본 확인

            # 정규화 좌표(0~1)인지 픽셀 좌표인지 자동 판단
            if max(raw) <= 1.0:
                x1 = int(raw[0] * w)
                y1 = int(raw[1] * h)
                x2 = int(raw[2] * w)
                y2 = int(raw[3] * h)
            else:
                x1, y1, x2, y2 = map(int, raw)

            name = CLASS_NAMES.get(cls, "unknown")
            print(f"      {name:<10}  conf={cf:.3f}  box=({x1},{y1})-({x2},{y2})")

        # 바운딩박스 시각화
        for box in boxes:
            cls  = int(box.cls[0])
            cf   = float(box.conf[0])
            raw  = box.xyxy[0].tolist()
            if max(raw) <= 1.0:
                x1 = int(raw[0] * w)
                y1 = int(raw[1] * h)
                x2 = int(raw[2] * w)
                y2 = int(raw[3] * h)
            else:
                x1, y1, x2, y2 = map(int, raw)
            color = COLORS.get(cls, (200, 200, 200))
            name  = CLASS_NAMES.get(cls, "?")

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {cf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    print(f"{'='*40}")
    print(f" 총 {len(boxes)}개 탐지")
    print(f"{'='*40}\n")

    # 결과 저장 + 화면 표시
    out_path = "test_result.jpg"
    cv2.imwrite(out_path, frame)
    print(f"결과 이미지 저장: {out_path}")

    cv2.imshow("Test Result (아무 키나 누르면 종료)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="ONNX 모델 탐지 테스트")
    parser.add_argument("--source", default="0", help="이미지 파일 경로 또는 웹캠(0)")
    parser.add_argument("--conf",   type=float, default=0.4, help="신뢰도 임계값 (기본: 0.4)")
    args = parser.parse_args()
    run(args.source, args.conf)


if __name__ == "__main__":
    main()
