"""
SafeWalk - ROI 기반 구역 탐지 및 TTS 경보 시스템
==============================================

보행 경로에 사다리꼴 ROI를 정의하고 3구역으로 나눠
교통약자에게 음성 경보를 제공합니다.

  Zone 1 (DANGER)  ── 바로 앞  (하단 1/3) : 즉시 위험 경보
  Zone 2 (CAUTION) ── 조금 먼  (중간 1/3) : 주의 경보
  Zone 3 (WATCH)   ── 먼 거리  (상단 1/3) : 선택적 경보 (--no-zone3 으로 비활성화)

사다리꼴 ROI 구조 (카메라 전방 시점):
         [top-left]──────[top-right]
              /    Zone3(먼)    \\
             /   Zone2(조금먼)   \\
            /    Zone1(바로앞)    \\
    [bottom-left]────────────[bottom-right]

사용법:
  python scripts/detect_roi.py                     # 웹캠 (기본)
  python scripts/detect_roi.py --source video.mp4  # 영상 파일
  python scripts/detect_roi.py --no-zone3          # Zone3 알림 비활성화
  python scripts/detect_roi.py --conf 0.45         # 신뢰도 임계값 조정
  python scripts/detect_roi.py --no-display        # 화면 없이 실행 (임베디드용)

TTS 설정:
  Korean TTS (Linux): sudo apt-get install espeak-ng
                      pip install pyttsx3
  Korean TTS (대안):  pip install gtts playsound
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] pip install ultralytics")
    sys.exit(1)


# ============================================================
# 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "safewalk_v1" / "weights" / "best.onnx"

CLASS_NAMES    = {0: "bicycle",  1: "kickboard",  2: "bollard"}
CLASS_NAMES_KO = {0: "자전거",   1: "킥보드",      2: "볼라드"}


# ============================================================
# ROI 및 구역 설정  ── 이 값만 수정해 ROI 크기/위치 조정
# ============================================================

# 사다리꼴 ROI 꼭짓점 (정규화 좌표, 0.0~1.0)
# 하단이 넓고 상단이 좁은 형태로 보행로를 표현
ROI_NORM = np.array([
    [0.35, 0.40],   # top-left
    [0.65, 0.40],   # top-right
    [0.95, 0.95],   # bottom-right
    [0.05, 0.95],   # bottom-left
], dtype=np.float32)

# ROI 내부 Y 분할 비율 (0=top, 1=bottom)
# Zone3 | Zone2 | Zone1 순서로 상→하
ZONE_Y_SPLITS = [0.0, 1 / 3, 2 / 3, 1.0]

# 구역 시각화 색상 (BGR)
ZONE_COLORS = {
    1: (0,   0,   255),   # DANGER  ── 빨강
    2: (0,   165, 255),   # CAUTION ── 주황
    3: (0,   255, 255),   # WATCH   ── 노랑
}

# 구역별 TTS 메시지 템플릿
ZONE_MESSAGES = {
    1: "{obj}가 바로 앞에 있습니다! 즉시 멈추세요!",
    2: "{obj} 주의하세요.",
    3: "전방에 {obj}가 있습니다.",
}

# 구역별 경보 쿨다운 (초) ── 중복 경보 방지
ZONE_COOLDOWN = {1: 1.5, 2: 3.0, 3: 5.0}


# ============================================================
# 좌표 계산 유틸
# ============================================================

def scale_roi(norm_pts: np.ndarray, w: int, h: int) -> np.ndarray:
    """정규화 좌표 → 픽셀 좌표"""
    pts = norm_pts.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts.astype(np.int32)


def get_zone_polygons(roi_pts: np.ndarray) -> dict[int, np.ndarray]:
    """
    사다리꼴 ROI를 3개 구역 폴리곤으로 분할.

    roi_pts 인덱스:
      0: top-left   1: top-right
      3: bot-left   2: bot-right
    """
    tl, tr, br, bl = (roi_pts[0].astype(float), roi_pts[1].astype(float),
                      roi_pts[2].astype(float), roi_pts[3].astype(float))

    def lerp(p1, p2, t):
        return p1 + t * (p2 - p1)

    zones: dict[int, np.ndarray] = {}
    for zone_id in range(1, 4):
        # Zone1=하단, Zone2=중간, Zone3=상단
        t_top = ZONE_Y_SPLITS[3 - zone_id]
        t_bot = ZONE_Y_SPLITS[4 - zone_id]

        lt = lerp(tl, bl, t_top)   # left-top corner of this zone
        lb = lerp(tl, bl, t_bot)   # left-bottom corner
        rt = lerp(tr, br, t_top)   # right-top corner
        rb = lerp(tr, br, t_bot)   # right-bottom corner

        zones[zone_id] = np.array([lt, rt, rb, lb], dtype=np.int32)

    return zones


def detect_zone(cx: int, cy: int, zones: dict[int, np.ndarray]) -> int:
    """점(cx, cy)이 속한 구역 번호 반환. ROI 바깥이면 0."""
    for zone_id in (1, 2, 3):   # 가까운 구역 우선
        result = cv2.pointPolygonTest(zones[zone_id], (float(cx), float(cy)), False)
        if result >= 0:
            return zone_id
    return 0


# ============================================================
# TTS 엔진
# ============================================================

class TTSEngine:
    """
    백그라운드 스레드로 TTS를 처리해 탐지 루프를 블로킹하지 않음.

    우선순위 큐: Zone1 경보가 대기 중인 Zone2/3 경보를 밀어냄.
    """

    def __init__(self):
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=10)
        self._backend = self._init_backend()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ----------------------------------------------------------
    def _init_backend(self):
        """pyttsx3 → gtts+pygame → 콘솔 출력 순으로 시도"""
        # 1) pyttsx3
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            for v in voices:
                if "korean" in v.id.lower() or "ko_" in v.id.lower():
                    engine.setProperty("voice", v.id)
                    break
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 1.0)
            print("[TTS] Backend: pyttsx3")
            return ("pyttsx3", engine)
        except Exception:
            pass

        # 2) gTTS + pygame
        try:
            import io
            import gtts
            import pygame
            pygame.mixer.init()
            print("[TTS] Backend: gTTS + pygame")
            return ("gtts", None)
        except Exception:
            pass

        # 3) 콘솔 출력 (TTS 없이)
        print("[TTS] Backend: console (TTS 라이브러리 없음 → 콘솔 출력으로 대체)")
        print("      pip install pyttsx3  또는  pip install gtts pygame")
        return ("console", None)

    # ----------------------------------------------------------
    def _worker(self):
        while True:
            try:
                priority, text = self._queue.get(timeout=1)
            except queue.Empty:
                continue
            if text is None:
                break
            self._say(text)

    def _say(self, text: str):
        backend_type = self._backend[0]

        if backend_type == "pyttsx3":
            try:
                engine = self._backend[1]
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS] pyttsx3 error: {e}")

        elif backend_type == "gtts":
            try:
                import io
                import gtts
                import pygame
                tts_obj = gtts.gTTS(text=text, lang="ko")
                fp = io.BytesIO()
                tts_obj.write_to_fp(fp)
                fp.seek(0)
                pygame.mixer.music.load(fp)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
            except Exception as e:
                print(f"[TTS] gTTS error: {e}")

        else:
            print(f"[TTS] {text}")

    # ----------------------------------------------------------
    def speak(self, text: str, priority: int = 2):
        """
        priority: 1=최고(Zone1), 2=중간(Zone2), 3=낮음(Zone3)
        낮은 숫자가 먼저 처리됨 (PriorityQueue 특성)
        """
        try:
            # Zone1 경보 시 대기 큐를 비워 즉시 발화
            if priority == 1:
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
            self._queue.put_nowait((priority, text))
        except queue.Full:
            pass

    def stop(self):
        self._queue.put((0, None))


# ============================================================
# 경보 쿨다운 관리
# ============================================================

class AlertManager:
    """
    구역 진입 시 1회만 경보. 객체가 구역을 벗어났다가 다시 들어와야 재경보.
    Zone3는 --no-zone3 옵션으로 비활성화 가능.
    """

    def __init__(self, zone3_enabled: bool = True):
        self.zone3_enabled = zone3_enabled
        self._prev_zones: set[int] = set()   # 직전 프레임의 점유 구역

    def update(self, current_zones: set[int]) -> list[int]:
        """
        current_zones: 이번 프레임에 탐지된 구역 집합
        반환: 이번에 새로 진입한 구역 목록 (경보 대상)
        """
        if not self.zone3_enabled:
            current_zones = current_zones - {3}

        newly_entered = sorted(current_zones - self._prev_zones)  # 이전엔 없었던 구역
        self._prev_zones = current_zones
        return newly_entered

    @staticmethod
    def build_message(zone: int, class_id: int) -> str:
        obj = CLASS_NAMES_KO.get(class_id, "장애물")
        return ZONE_MESSAGES[zone].format(obj=obj)


# ============================================================
# 시각화 헬퍼
# ============================================================

_ZONE_LABELS = {1: "Z1-DANGER", 2: "Z2-CAUTION", 3: "Z3-WATCH"}


def draw_zones(frame: np.ndarray, zones: dict[int, np.ndarray], alpha: float = 0.18):
    overlay = frame.copy()
    for zone_id, poly in zones.items():
        cv2.fillPoly(overlay, [poly], ZONE_COLORS[zone_id])
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for zone_id, poly in zones.items():
        cv2.polylines(frame, [poly], True, ZONE_COLORS[zone_id], 1, cv2.LINE_AA)
        cx = int(poly[:, 0].mean())
        cy = int(poly[:, 1].mean())
        cv2.putText(frame, _ZONE_LABELS[zone_id],
                    (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    ZONE_COLORS[zone_id], 1, cv2.LINE_AA)


def draw_detection(frame: np.ndarray, xyxy: tuple, class_id: int,
                   conf: float, zone: int):
    x1, y1, x2, y2 = map(int, xyxy)
    color = ZONE_COLORS.get(zone, (180, 180, 180))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{CLASS_NAMES.get(class_id, '?')} {conf:.2f}"
    if zone > 0:
        label += f" [{_ZONE_LABELS[zone]}]"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw + 3, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, zone3_on: bool, fps: float):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (230, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Zone3: {'ON' if zone3_on else 'OFF'}", (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255) if zone3_on else (100, 100, 100), 1)


# ============================================================
# 메인 탐지 루프
# ============================================================

def run(source: str, conf_thresh: float, zone3_enabled: bool, display: bool):
    if not MODEL_PATH.exists():
        print(f"[ERROR] 모델 파일 없음: {MODEL_PATH}")
        print("  → python scripts/train.py train  으로 먼저 학습하세요")
        sys.exit(1)

    print(f"[INFO] 모델 로드: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH), task="detect")

    cap_src = 0 if source == "0" else source
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print(f"[ERROR] 소스를 열 수 없음: {source}")
        sys.exit(1)

    tts   = TTSEngine()
    alert = AlertManager(zone3_enabled=zone3_enabled)

    print(f"[INFO] 신뢰도 임계값 : {conf_thresh}")
    print(f"[INFO] Zone3 알림    : {'활성화' if zone3_enabled else '비활성화'}")
    print("[INFO] 탐지 시작. 'q' 키로 종료.")

    zones_cache: dict | None = None
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # ── 구역 폴리곤 (해상도 변경 시만 재계산) ──────────────────
            if zones_cache is None or zones_cache["size"] != (w, h):
                roi_px = scale_roi(ROI_NORM, w, h)
                zones  = get_zone_polygons(roi_px)
                zones_cache = {"size": (w, h), "zones": zones}
            else:
                zones = zones_cache["zones"]

            # ── YOLO 추론 ────────────────────────────────────────────
            results = model(frame, conf=conf_thresh, verbose=False)[0]

            # ── 구역 오버레이 ─────────────────────────────────────────
            if display:
                draw_zones(frame, zones)

            # ── 구역별 최고 신뢰도 탐지 수집 ─────────────────────────
            # key: zone_id, value: (class_id, conf, xyxy)
            zone_best: dict[int, tuple] = {}

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0])
                conf     = float(box.conf[0])

                # 바운딩박스 하단 중심 (객체 지면 접점) 기준으로 구역 판단
                cx = int((x1 + x2) / 2)
                cy = int(y2)
                zone = detect_zone(cx, cy, zones)

                if display:
                    draw_detection(frame, (x1, y1, x2, y2), class_id, conf, zone)

                if zone > 0:
                    if zone not in zone_best or conf > zone_best[zone][1]:
                        zone_best[zone] = (class_id, conf, (x1, y1, x2, y2))

            # ── 경보 발생: 새로 진입한 구역만 1회 ────────────────────────
            newly_entered = alert.update(set(zone_best.keys()))
            for zone_id in newly_entered:
                class_id, _, _ = zone_best[zone_id]
                msg = alert.build_message(zone_id, class_id)
                tts.speak(msg, priority=zone_id)
                print(f"[ALERT Zone{zone_id}] {msg}")

            # ── FPS / HUD ────────────────────────────────────────────
            now  = time.time()
            fps  = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            if display:
                draw_hud(frame, zone3_enabled, fps)
                cv2.imshow("SafeWalk ROI Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        tts.stop()
        if display:
            cv2.destroyAllWindows()
        print("[INFO] 탐지 종료.")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SafeWalk ROI 기반 구역 탐지 + TTS 경보",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
구역 정의 (사다리꼴 ROI 내부):
  Zone 1  DANGER  (하단 1/3) ── 바로 앞, 즉시 경보, 쿨다운 1.5s
  Zone 2  CAUTION (중간 1/3) ── 조금 먼, 주의 경보, 쿨다운 3.0s
  Zone 3  WATCH   (상단 1/3) ── 먼 거리, 선택적 경보, 쿨다운 5.0s

예시:
  python scripts/detect_roi.py
  python scripts/detect_roi.py --source video.mp4
  python scripts/detect_roi.py --no-zone3
  python scripts/detect_roi.py --conf 0.45 --no-display
        """,
    )
    parser.add_argument(
        "--source", default="0",
        help="입력 소스: 웹캠 인덱스(0,1,...) 또는 영상 파일 경로 (기본: 0)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.40,
        help="탐지 신뢰도 임계값 0~1 (기본: 0.40)",
    )
    parser.add_argument(
        "--no-zone3", action="store_true",
        help="Zone3 (먼 거리) TTS 알림 비활성화",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="화면 출력 없이 실행 (서버 / 임베디드 환경용)",
    )
    args = parser.parse_args()

    run(
        source=args.source,
        conf_thresh=args.conf,
        zone3_enabled=not args.no_zone3,
        display=not args.no_display,
    )


if __name__ == "__main__":
    main()
