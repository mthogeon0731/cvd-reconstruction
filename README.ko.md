# CVD/CBD 재구현 0.2.2

연구 설명과 제공 명세에 기반한 **새 구현**이며, 합성 데이터로 소프트웨어를 검증한 연구용 프로토타입입니다. 원본 연구 코드 복구본이나 실측 성능 재현본이 아닙니다. 실제 연구 원자료·개인자료·과거 출력·학습 모델은 포함하지 않습니다.

구현 범위는 정상상태 1D 반응–확산 물리, 평판 성장·점도 교정, TIFF REF/DARK 보정과 등록 ROI 특징, 조건별 nested RF/KRR/SVR 평가, 별도 최종 모델, bot/top SC, 독립 물성 기반 이론 비교, 불확도 전파와 모델상 AR 한계입니다. [가정과 한계](docs/MODEL.md)를 먼저 확인하세요.

## 설치와 실행

저장소 루트에서 새 환경을 만듭니다. 패키지는 Python >=3.10을 선언하지만, 이번에 실행한 조합은 [검증 기록](docs/VALIDATION.md)에 한정됩니다. lock의 의존성이 더 높은 Python 버전을 요구할 수 있습니다.

```sh
python -m venv .venv
```

Windows PowerShell은 `.venv\Scripts\Activate.ps1`, Linux/macOS는 `source .venv/bin/activate`로 활성화한 뒤 실행합니다.

```sh
python -m pip install -r requirements-build-lock.txt
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps .
python -m pip check
python examples/minimal.py
python -m pytest tests -q
python scripts/check_repository.py
python -m cvd_cbd study generate --config configs/synthetic_study.json --root outputs/generated_inputs
python -m cvd_cbd study validate --config configs/synthetic_study.json --root outputs/generated_inputs
python -m cvd_cbd study demo --config configs/synthetic_study.json --out outputs/demo
python verification/verify_run.py --run outputs/demo --out outputs/demo_verification.json
```

기본 제품 테스트는 임시 폴더에서 합성 입력과 모델을 생성합니다. 과거 감사 폴더·외부 fixture·기존 가상환경에 의존하지 않습니다. 모든 명령은 새 출력 경로를 사용합니다. [단계별 14개 CLI 명령](docs/CLI.md)도 저장소의 생성기로 만든 자료를 사용합니다.

## 검증 범위

합성 생성기는 난수·물리식으로 TIFF, 물성, FE-SEM 대용 라벨과 가상의 교정 증빙을 만듭니다. 합성 R² 약 0.9967과 상사성 R² 약 0.99989는 과거 소프트웨어 일관성 지표입니다. **같은 물리식으로 생성·비교하므로 실험 성능이나 독립적인 물리 검증을 입증하지 않습니다.** 실측 OM·FE-SEM 성능 검증은 NOT RUN입니다.

과거 505개 전체 테스트(제품+adapter 감사), 독립 89개, README 14단계는 이번 실행 결과와 구분합니다. 원래 감사의 **144 PASS / 1 mutation 호환성 FAIL**도 숨기지 않고 [외부 감사 기록](docs/EXTERNAL_AUDITS.md)에 설명합니다. 이번 명령·판정은 [현재 검증 기록](docs/VALIDATION.md)에 있습니다.

## 실제 데이터 준비

[templates](templates)의 CSV는 헤더만 포함합니다. 자료는 Git 밖에서 관리하고, [입력 계약](docs/INPUT_CONTRACTS.md)에 따라 작성하세요. `configs/real_study_template.json`의 수치는 미확인 자리표시자이며 관측창 수정 없이 실행 가능한 실측 예제가 아닙니다.

실제 TIFF 3장부터 REF/DARK·촬영 조건·스테이지 교정·ROI·관찰 z와 라벨 대응을 검토해야 합니다. 명목 폭으로 픽셀 스케일을 맞추거나, 타일에 하나의 라벨을 복제해 독립 표본으로 세면 안 됩니다. 해시나 파일 존재 검사는 측정 진실성을 인증하지 않습니다. 최종 모델의 학습자료 재예측도 독립 성능이 아닙니다. 모델은 신뢰한 출처에서 직접 생성한 경우에만 `--trust-model`로 로드하세요.

이번 사본의 [변경 기록](CHANGELOG.md), [출처·보안 점검 범위](docs/PROVENANCE.md), [권리와 라이선스 상태](docs/RIGHTS.md)를 확인하세요. 프로젝트 라이선스는 미선택 상태입니다. 공개 전환은 소유자의 별도 승인을 기다립니다.
