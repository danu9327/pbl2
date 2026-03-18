교통약자를 위한 AI 음성 보행 내비게이션

## 데이터셋

| Source | Images | Classes | License |
|--------|--------|---------|---------|
| [bicycle (Dng)](https://universe.roboflow.com/dng-cjryd/bicycle-i7nhz) | 127 | Bicycle | CC BY 4.0 |
| [kickboard (Inha Univ)](https://universe.roboflow.com/inha-univ-vgzgz/kickboard-ibhkj) | 462 | kb | CC BY 4.0 |
| [bollard (project-60htx)](https://universe.roboflow.com/project-60htx/bollard-v2gn5) | 634 | bollad1, bollard, bollard_abnormal, bollard_normal, tubular marker_normal | CC BY 4.0 |
| **Total** | **1,223** | → **3 unified classes** | |

### Unified Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | bicycle | 자전거 |
| 1 | kickboard | 전동킥보드 |
| 2 | bollard | 볼라드 (5개 서브클래스 통합) |

---

## Quick Start

### 1. 환경 설정

```bash
pip install roboflow albumentations opencv-python-headless ultralytics
```

### 2. 데이터 다운로드

```bash
python scripts/data_pipeline.py download --api-key YOUR_ROBOFLOW_API_KEY
```

> Roboflow API key: https://app.roboflow.com/settings/api 에서 발급

### 3. 전체 파이프라인 실행 (merge → split → validate)

```bash
python scripts/data_pipeline.py all
```

### 4. 클래스 밸런싱 (bicycle 증강)

```bash
python scripts/augmentation.py --target-per-class 500
```

### 5. YOLO 학습 (Phase 2)

```bash
scripts/train.py 참고바람
```

---

## 프로젝트 구조

```
safewalk-nav/
├── configs/
│   └── dataset.yaml              # YOLO 설정 (3 classes)
├── scripts/
│   ├── data_pipeline.py          # 다운로드/리매핑/분할/검증
│   └── augmentation.py           # 클래스 밸런싱 증강
├── data/
│   ├── raw/                      # Roboflow 원본
│   │   ├── bicycle/
│   │   ├── kickboard/
│   │   └── bollard/
│   ├── merged/                   # 리매핑 후 통합
│   └── processed/                # 최종 train/val/test
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
└── models/                       # 학습된 weights
```

---

## 추론 결과

### Evaluation Results

mAP@0.5:      0.9274
 
mAP@0.5:0.95: 0.7064

Per-class AP@0.5:

bicycle      0.8802

kickboard    0.9540

bollard      0.9482

<img src="./assets/confusion_matrix_normalized.png" width="500">

## 개별 명령어

```bash
# 데이터셋 다운로드
python scripts/data_pipeline.py download --api-key YOUR_KEY

# 클래스 리매핑 + 통합
python scripts/data_pipeline.py merge

# Train/Val/Test 분할 (7:2:1)
python scripts/data_pipeline.py split --ratios 0.7,0.2,0.1 --seed 42

# 데이터 검증 + 통계
python scripts/data_pipeline.py validate

# 부족 클래스 자동 증강
python scripts/augmentation.py --target-per-class 500

# 특정 클래스만 증강
python scripts/augmentation.py --classes 0 --multiply 4
```

## Flutter & Server 아키텍처

## 1. 실시간 데이터 흐름도 (Data Flow Diagram)

| 단계 | 주체 (Location) | 작업 내용 (Action) | 데이터 형태 |
| :-- | :--- | :--- | :--- |
| **Step 1** | **Flutter App** | 카메라 영상 캡처 (5 FPS) | Raw Frame |
| **Step 2** | **Network** | WebSocket을 통한 실시간 스트리밍 | Binary (JPEG) |
| **Step 3** | **AI Server** | 데이터 수신 및 이미지 디코딩 | NumPy Array |
| **Step 4** | **AI Server** | **병렬 AI 추론 실행**<br>1. YOLOv10 (장애물 탐지)<br>2. Mediapipe (자세 분석) | Inference Results |
| **Step 5** | **AI Server** | 위험도 종합 평가 및 메시지 생성 | JSON Data |
| **Step 6** | **Flutter App** | 안내 실행 (TTS 음성 및 UI 업데이트) | Voice / UI |

---

## 2. 모듈별 상세 역할

### 📱 Flutter Client
* **Camera Module**: 후면 카메라를 통해 초당 5개($5\text{ FPS}$)의 프레임 추출.
* **WebSocket Service**: 서버와 영구적인 파이프라인 형성, 바이너리(Bytes) 데이터 송신.
* **TTS Engine**: 서버의 위험 메시지를 음성으로 변환 (위험 단계별 발화 우선순위 관리).
* **Overlay UI**: 카메라 화면 위에 실시간 위험 수치 및 상태바 렌더링.

### 🖥️ AI Analysis Server
* **FastAPI Wrapper**: 고성능 비동기 I/O를 통한 다중 클라이언트 대응.
* **YOLOv10 Engine**: $COCO$ 데이터셋 기반 17개 위험 클래스 선별 탐지.
* **Pose Engine**: 사용자의 3D 랜드마크를 추출하여 상체 기울기($Trunk\ Lean$) 분석.
* **Risk Analyzer**: 
    - 객체의 면적 비율($Area\ Ratio$)을 거리로 환산.
    - 객체 가중치와 자세 점수를 합산하여 최종 위험 점수 산출.

---

## 3. 분석 알고리즘 (Risk Scoring)

최종 위험 점수($S$)는 다음과 같은 가중 합산 방식을 따릅니다.

$$S = (D \times W) + (P \times 0.3)$$

- $D$ (Distance): 객체가 화면에서 차지하는 면적 비율
- $W$ (Weight): 객체별 위험 가중치 (예: 대형차=1.0, 벤치=0.5)
- $P$ (Posture): 보행 불안정 및 낙상 위험 수치

---

## 4. 위험 단계 정의 (Risk Levels)

| 단계 | 위험 점수 | 음성 안내 주기 | 주요 안내 내용 |
| :--- | :--- | :--- | :--- |
| **안전 (Safe)** | $0.0 \sim 0.3$ | 5초 | "전방 안전합니다." |
| **주의 (Caution)** | $0.3 \sim 0.6$ | 3초 | "왼쪽에 차량이 있습니다." |
| **위험 (Danger)** | $0.6 \sim 0.85$ | 1.5초 | "위험! 정면에 장애물 접근 중." |
| **긴급 (Critical)** | $0.85 \sim 1.0$ | 즉시 | "즉시 멈추세요! 낙상 위험." |

---
