# cvd-reconstruction · 0.2.2

**트렌치 코팅의 반응–확산 물리, 현미경 영상 특징, 조건 단위 ML을 연결해 두께와 스텝 커버리지를 추정합니다.**

![검증 상태: 합성자료만 검증](docs/assets/status.svg)

제공된 연구 설명과 명세를 바탕으로 새로 구현한 연구용 프로토타입입니다. 예제의 **영상·라벨·교정 입력은 모두 합성자료**입니다. 원본 연구실 코드 복구본이나 실측 성능 재현본이 아니며 실제 OM / FE-SEM 검증은 미실행입니다.

[빠른 실행](#quickstart) · [예제 결과](#example-results) · [모델](docs/MODEL.md) · [입력 계약](docs/INPUT_CONTRACTS.md) · [검증 기록](docs/VALIDATION.md) · [English](README.md)

![입력, 광학·물리 처리, 두께·SC·불확도 출력 흐름](docs/assets/pipeline.svg)

## Overview

직사각형 트렌치를 따라 코팅 두께가 어떻게 변하며, 상단 대비 하단에 얼마나 코팅되는지를 다룹니다. 정상상태 반응–확산 모델, 등록된 광학 관측과 두께 라벨을 연결하고 **학습에서 제외한 조건**에서 RF·KRR·SVR을 평가합니다.

결과물은 입력 생성, 영상 교정, ROI 특징, 평가 기록, 짝지은 SC, 이론 비교, 불확도 요약을 추적할 수 있는 Python 워크플로입니다. 전체 라벨로 별도 학습한 최종 모델은 추론용이며 학습자료 재예측을 독립 성능으로 보고하지 않습니다. 예제의 물성 교정도 합성입니다.

## Pipeline

| 단계 | 실제 구현 | 코드 |
|---|---|---|
| 물리 교정 | 평판 성장·점도로 반응/확산 입력 구성, 형상·재료별 속도로 정상 1D 수송 문제 정의 | [교정](cvd_cbd/calibration.py), [물리](cvd_cbd/physics.py) |
| 영상 계측 | 등록 TIFF의 REF/DARK 보정, `T=(I−DARK)/(REF−DARK)`, OD, 스케일·관측창 검사와 별도 텍스처 특징 | [계측](cvd_cbd/metrology.py), [특징](cvd_cbd/dataset.py) |
| 두께 평가 | 바깥 조건 LOGO, 안쪽 그룹 CV로 RF/KRR/SVR·하이퍼파라미터 선택, fold 내부 X/y 스케일링 | [ML](cvd_cbd/modeling.py) |
| 두께 → SC | 하단/상단 대응, 가능한 측정오차 전달, 예측 또는 라벨 SC를 동일 관측창의 고정 이론과 비교 | [추론](cvd_cbd/infer.py), [상사성](cvd_cbd/similitude.py) |
| 불확도 | 별도 입력한 평균·로그 공분산·측정 가정을 SC와 모델상 AR 한계로 Monte Carlo 전달 | [불확도](cvd_cbd/uncertainty.py) |

독립 라벨 단위는 ROI이며 타일은 추가 독립 표본이 아닙니다. 조건 사이의 배치·시편·영상·동일 콘텐츠 의존성은 거부합니다. Monte Carlo 구간은 주어진 가정에 조건부이며 교정된 ML 예측구간이 아닙니다. [가정과 경계](docs/MODEL.md).

## Quickstart

실행 환경은 **Windows 11 / CPython 3.12.14**와 고정 의존성입니다. 패키지는 Python ≥3.10을 선언하지만 lock과 모든 Python·OS 조합의 지원을 검증하지 않았습니다. 비공개 저장소를 clone하려면 권한이 필요합니다.

```sh
git clone https://github.com/mthogeon0731/cvd-reconstruction.git
cd cvd-reconstruction
python -m venv .venv
```

| 셸 | 활성화 |
|---|---|
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| macOS / Linux 문법 예시 — 해당 OS 검증 주장 아님 | `source .venv/bin/activate` |

```sh
python -m pip install -r requirements-build-lock.txt
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps .
python -m pip check
python examples/minimal.py
```

최소 예제는 입력 파일 없이 합성 끝점 농도비 `SC ≈ 0.87327137`을 출력합니다. ML 학습이나 실험 검증은 하지 않습니다.

전체 합성 연구와 README 그림 생성:

```sh
python -m cvd_cbd study demo --config configs/synthetic_study.json --out outputs/portfolio_demo
python verification/verify_run.py --run outputs/portfolio_demo --out outputs/portfolio_demo_verification.json
python scripts/render_example.py --run outputs/portfolio_demo --out docs/assets
```

연구 실행마다 **새 출력 경로**를 사용하세요. 첫 명령이 입력을 직접 생성하므로 외부 감사 자료가 필요하지 않습니다. `cvd-study`는 `python -m cvd_cbd study`와 같은 설치 진입점입니다. 예전 `python demo.py`는 별도의 미교정 surrogate 경로입니다. 입력을 따로 생성하고 단계별로 실행하려면 [14개 CLI 명령](docs/CLI.md)을 참고하세요.

## Example Results

![Synthetic example: 제외된 조건의 두께 예측과 교차 예측 SC의 고정 이론 비교](docs/assets/synthetic-results.png)

*Synthetic example, seed 0. A의 색은 6개 조건을 구분하며 모든 점은 바깥 fold 예측입니다. B는 그 예측의 조건/AR 평균 SC와 실제 관측창의 이론을 비교합니다. 연구 사진이나 실측 자료는 사용하지 않았습니다.*

| 이번에 새로 생성한 합성 예제 | 결과 | 의미 |
|---|---:|---|
| 트렌치 라벨 ROI / 조건 | 144 / 6 | 타일 수와 구분 |
| Nested 두께 R² / RMSE | 0.996754 / 2.657990 nm | 제외 조건의 합성 라벨 예측 |
| 교차 예측 SC의 이론 비교 R² | 0.998786 | 예측 두께에서 구한 조건/AR 36점 |
| 합성 라벨 SC의 이론 비교 R² | 0.999887 | FE-SEM 대용 합성 라벨, 실측 아님 |

생성기는 `OD = thickness_nm / 450`, 가상 카메라·라벨 잡음, 이론 비교와 같은 물리식을 씁니다. 높은 수치는 **공통 생성기와의 소프트웨어 일관성**이며 실험 정확도나 독립 물리 검증이 아닙니다. 과거 약 0.9967·0.99989도 같은 한계 아래 해석해야 합니다. [그림 입력·설정·정확한 지표](docs/EXAMPLE.md).

## Data & Outputs

입력은 [생성 코드](cvd_cbd/synthetic.py)와 [설정](configs/synthetic_study.json)에서 만듭니다. 실측 자료는 배포하지 않습니다. [templates](templates)의 CSV는 헤더만 있고, [입력 계약](docs/INPUT_CONTRACTS.md)에 필요한 증거를 정의했습니다. 실측 설정에는 미확인 자리표시자가 있어 그대로 실행 가능한 예제가 아닙니다.

아래 경로는 `outputs/portfolio_demo/` 기준입니다.

| 경로 | 내용 |
|---|---|
| `SYNTHETIC_inputs/` | 생성 TIFF, 라벨, 물성표, 가상 교정 증빙 |
| `SYNTHETIC_results/features/` | 등록 ROI 특징과 영상 QC |
| `SYNTHETIC_results/evaluation/` | 분할·안쪽 후보·선택된 교차 예측·지표 |
| `SYNTHETIC_results/final_model/` | 전체 라벨로 별도 학습한 추론 모델 |
| `SYNTHETIC_results/similitude_cross_fitted/` | 예측 SC와 동일 관측창 이론 비교 |
| `SYNTHETIC_results/similitude_fesem/` | 합성 라벨 SC의 이론 비교 |
| `SYNTHETIC_results/process_window/` | 모델상 AR 한계 |
| `SYNTHETIC_results/uncertainty_report.json` | 조건부 Monte Carlo 요약 |

대량 영상·모델·출력은 Git에서 제외하고 대표 그림과 경로를 정리한 요약만 추적합니다. 실행 provenance에는 로컬 경로가 생길 수 있어 공유 전 확인해야 합니다. joblib는 역직렬화 중 코드가 실행될 수 있으므로 신뢰하는 모델만 읽으세요.

## Evaluation

```sh
python -m pytest tests -q
python scripts/check_repository.py
```

저장소 제품 테스트는 **360개**이며 JSON 회귀 49개를 포함하고 임시 입력을 직접 생성합니다. 이번 명령·실행 결과는 [검증 기록](docs/VALIDATION.md)에 있으며 그림 검사는 제품 테스트 수에 합산하지 않습니다.

과거 **505개**는 제품 360 + adapter 감사 145이고, 독립 **89개**는 별도 과거 결과입니다. 원본 감사의 **144 PASS / 1 mutation 호환성 FAIL**을 유지합니다. README **14단계 실행**과 별도 **14개 smoke 검사**도 다른 절차이며 서로 대체하거나 이번 결과에 합산하지 않습니다. [감사 범위·원본 실패·외부 재실행 조건](docs/EXTERNAL_AUDITS.md).

## Reproducibility

Seed **0**, 커밋된 설정과 고정 의존성을 사용합니다. [재현 방법](docs/REPRODUCIBILITY.md)에 입력 위치·비교 파일·명령·환경 제어를, [요약 JSON](docs/assets/synthetic-summary.json)에 사적인 경로 없이 설정·해시·지표를 기록했습니다.

동일 환경의 바이트 일치와 OS 간 수치 일치는 다른 검사입니다. 과거 동일 환경 일치 및 Windows/Linux 차이는 과거 기록으로 유지합니다. 이번에 Linux·macOS·다른 Python·GitHub Actions가 통과했다고 주장하지 않습니다.

## Limitations

- **실측 OM / FE-SEM 미검증.** 광학 변환, 스케일 증빙, 대응 라벨, 촬영 독립성과 물성 교정을 실험으로 확인해야 합니다.
- **제한된 물리.** 정상·희석 1D 수송, 일정 확산, 1차 표면반응 가정이며 형상 변화·유동·과도 핵생성·기체/액체 증착 동등성을 검증하지 않습니다.
- **조건부 불확도.** 입력 분포와 고정 교차 예측의 조건 bootstrap은 전체 모델 선택 및 미지의 계통오차를 포함하지 않습니다.
- **합성 시간 외삽.** 10–40분 평판 성장 자료를 2시간 영상 예제로 외삽하며 실측 시간 안정성은 미검증입니다.
- **모델상 AR 한계.** 선택한 가정·임곗값의 결과이며 제조 보증이나 보편적 임계 AR이 아닙니다.

## Repository Structure

```text
cvd_cbd/        물리·계측·ML·study CLI
configs/        합성 설정과 미검증 실측 템플릿
templates/      빈 스키마와 불확도 템플릿
examples/       결정론적 최소 물리 예제
scripts/        그림 생성과 저장소·CLI 검사
tests/          자체 실행 가능한 제품 테스트
verification/   출력·반복 실행 비교
docs/           모델·입력 계약·검증·권리·대표 그림
```

## References

- [모델 수식·관측량·가정](docs/MODEL.md)과 [Pipeline](#pipeline)의 실제 코드.
- [scikit-learn 그룹 교차검증](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
- [출처 기록](docs/PROVENANCE.md): 제공 설명·명세에 기반한 재구현입니다. 원본 연구실 코드 및 독립 확인된 논문 서지는 배포하지 않으며 과거 연구 성능을 현재 결과로 주장하지 않습니다.

연구에 활용한다면 저장소와 사용한 커밋·설정을 표시해 주세요. 이는 자발적 재현성 요청이며 MIT에 추가된 의무가 아닙니다.

## License

[MIT](LICENSE) · Copyright (c) 2026 Hogeon Kim. 소유자 권한의 코드·테스트·문서·합성 그림에 적용하며 외부 의존성의 라이선스는 유지합니다. [권리와 적용 범위](docs/RIGHTS.md), [의존성 고지](docs/DEPENDENCIES.md).

라이선스 승인은 공개 전환 승인이 아닙니다. 소유자가 최종 커밋을 별도 승인할 때까지 비공개 상태를 유지합니다.
