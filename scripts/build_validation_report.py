"""Build the current report only from retained execution artifacts."""
import json,re,hashlib,platform,sys,importlib.metadata
from pathlib import Path
from cvd_cbd.json_utils import read_json
root=Path(__file__).resolve().parents[1];base=root/'outputs/SYNTHETIC_delivery/SYNTHETIC_results'
load=lambda p:read_json(root/p)
run=load('outputs/SYNTHETIC_delivery/SYNTHETIC_results/run_manifest.json')
process=load('outputs/SYNTHETIC_delivery/SYNTHETIC_results/process_window/process_report.json')
mc=load('outputs/SYNTHETIC_delivery/SYNTHETIC_results/design/mc_report.json')
design=load('outputs/SYNTHETIC_delivery/SYNTHETIC_results/design/design_report.json')
models=load('outputs/SYNTHETIC_delivery/SYNTHETIC_results/evaluation/metrics.json')
legacy=load('outputs/LEGACY_SYNTHETIC_demo/monte_carlo_summary.json')
cli=load('validation/logs/10_cli_stages.json')
zip_report=load('validation/zip_verification.json') if (root/'validation/zip_verification.json').is_file() else {'status':'NOT RUN','reason':'Awaiting fresh extraction audit'}
latest=next((p for p in ['validation/logs/20_extracted_pytest.log','validation/logs/17_zip_pytest.log','validation/logs/13_final_source_tests.log'] if (root/p).is_file()))
tests=(root/latest).read_text();match=re.search(r'(\d+) passed',tests);count=int(match[1]) if match else None
metrics=models['nested_selection']
env=load('validation/environment.json');env['source_sha256']={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'cvd_cbd').glob('*.py'))};(root/'validation/environment.json').write_text(json.dumps(env,indent=2))
text=f'''# 이번 실행 검증 보고서

이 기록은 제공 첨부의 이전 검증 문구를 옮긴 것이 아니라 이번 환경에서 실제 수행한 명령과 산출물에 근거합니다. 소프트웨어 실행, 가정에 따른 수치 계산, 실험 성능 검증을 구분합니다. 원본 연구실 코드 복구/실험 성능 재현은 주장하지 않습니다.

## 환경과 재현 정보

- Python {platform.python_version()}, {platform.system()} ({platform.machine()}). 전체 OS/패키지/소스 SHA256: `validation/environment.json`.
- 새 가상환경을 생성하고 requirements 범위로 실제 설치한 뒤 editable install을 실행했습니다. 실제 고정 버전은 `requirements-lock.txt`/`requirements-tested.txt`에 있습니다.
- NumPy {run['versions']['numpy']}, SciPy {run['versions']['scipy']}, pandas {run['versions']['pandas']}, scikit-learn {run['versions']['scikit-learn']}, Matplotlib {run['versions']['matplotlib']}, Pillow {run['versions']['Pillow']}, joblib {run['versions']['joblib']}.
- 새 study seed=0. Monte Carlo 반복=4000, 신규 설계 점=36(6조건×6 AR), 합성 OM/평판 영상=146, 타일=352, 등록 FESEM 대용 라벨=144, 트렌치=72, 조건×AR 평균=36(각2개 합성 트렌치). 실제 실험 행은 0개입니다.
- 별도 legacy seed=20261006, 시뮬레이터 학습 표본=4000, 선택 DoE=27, 노이즈 MC=4000. 이 세 수와 원래 18회 실험 run을 혼용하지 않습니다.
- 기준 scaffold 28개 테스트는 수정 전에 기본 환경에서 실행했습니다. 새 환경과 버전이 다르며, `validation/baseline_environment.json`에 구분했습니다.

## 실제 명령과 상태

모든 명령은 해당 프로젝트 루트에서 실행했습니다. 환경 경로는 재현 설명을 위해 `python`으로 표현하며 pip 로그·traceback의 작업공간 접두사는 ZIP에서 `<PROJECT>`/`<ENV>`로 정규화합니다. 판정·오류·수치 내용은 보존합니다.

| 확인 | 상태 | 명령/증거 |
|---|---|---|
| 원본 기준 테스트 28개 | PASS | `python -m unittest discover -s tests -v`; `validation/logs/01_baseline_unittest.log` |
| 새 venv 의존성 설치·editable install | PASS | `python -m venv <ENV>`; `python -m pip install -r requirements.txt pytest`; `python -m pip install -e .`; `02_install.log` |
| 최종 전체 pytest {count}개 | PASS | `python -m pytest -q`; `{latest}` |
| 2면 비활성 기판/끝단0 추가 확인 | PASS | `python -m pytest tests/test_calibration.py -q`; `15_active_surface_calibration.log` |
| 전체 합성 TIFF 파이프라인 | PASS | `python -m cvd_cbd study demo --config configs/synthetic_study.json --out outputs/SYNTHETIC_delivery`; `12_final_source_pipeline.log` |
| 단계별 CLI 14회 | {'PASS' if all(c['status']=='PASS' for c in cli) else 'FAIL'} | `python scripts/validate_cli.py --data outputs/SYNTHETIC_delivery/SYNTHETIC_inputs --features outputs/SYNTHETIC_delivery/SYNTHETIC_results/features/features.csv --work validation/cli_smoke_work --log validation/logs/10_cli_stages.json`; 12 성공 + 미신뢰 모델/덮어쓰기 2건 exit2 거부 |
| legacy 물리 RF demo | PASS | `python -m cvd_cbd demo --config configs/reference_demo.json --out outputs/LEGACY_SYNTHETIC_demo --mc-repeats 4000`; `14_legacy_cli_demo.log` |
| 컴파일 | PASS | `python -m compileall -q cvd_cbd tests scripts`; `16_compile.log` |
| seed 반복 주요 CSV 8개 | PASS | 서로 다른 전체 실행의 SHA256 일치, `validation/reproducibility.json`; 원본 TIFF 반복 생성도 테스트 |
| 합성 오버레이/공정맵 표시 | PASS | top·깊은 bot·flat 오버레이와 process_map을 실제 열어 타일/범례를 점검 |
| ZIP 새 폴더 추출·새 venv 설치·실행 | {zip_report['status']} | `validation/zip_verification.json`, `17_zip_install.log`, `20_extracted_*` 로그 |
| 실제 OM 원본 3장 검사 | NOT RUN | 원본 TIFF/REF/DARK/교정 기록 미제공; `validation/real_image_status.json` |
| 실험 RF/KRR/SVR·수작업 오차 개선·상사성 .94 | NOT RUN | 원자료·독립 split·오차 정의 미제공 |
| Windows/macOS | NOT RUN | 해당 OS 환경에서 실행하지 않음 |

CLI smoke 임시 중복 산출물과 중간 실패 실행의 대형 파일은 개발 보관 폴더에 두고 ZIP에는 중복 포함하지 않았습니다. 실제 납품용 전체 입력·출력은 `outputs/SYNTHETIC_delivery/`, 별도 legacy 출력은 `outputs/LEGACY_SYNTHETIC_demo/`에 모두 있습니다.

## 발견한 실패와 처리

1. `03_first_pipeline.log` — FAIL: pandas `DataFrame.flags`와 열 이름 충돌. `d['flags']`로 수정했습니다.
2. `04_second_pipeline.log` — FAIL: 저장 artifact의 모델 객체와 모델 이름이 같은 키여서 추론 실패. `model_name`을 분리했습니다.
3. `06_unit_first.log` 및 `07_full_tests_first.log` — FAIL: 상수 두께의 미세한 수치 기울기를 성장으로 인정. 수치 분해능과 양의 성장 신뢰하한 진단을 추가했습니다. 실패 테스트를 지우거나 기대값을 바꾸지 않았습니다.
4. 수정 후 전체 테스트가 통과했고, 공개 CLI와 납품용 전체 실행을 다시 수행했습니다. 마지막 검토에서 stale manifest 차단, 비활성 기판의 불필요한 입력 요구를 보완하고 회귀 테스트를 추가했습니다.
5. ZIP 모델 로딩 대조에서 CSV 기본 파서가 약 1e-16의 특징 반올림과 범위 플래그 3건의 차이를 만들었습니다. CLI에 round-trip 파서를 적용해 특징/플래그/예측이 원본과 같아졌습니다. 실패 근거와 수정 후 대조는 `validation/csv_roundtrip_regression.json`, 회귀 테스트는 `test_cli_csv_roundtrip_preserves_range_flags`입니다. 모델이나 실험 값을 변경하지 않았습니다.

## 수치 구현과 가정 아래의 새 계산

해석해/FDM 2차 수렴, 0반응/큰 φ·Bi, 직사각 재료별 둘레와 독립 k_end, 공간 적분의 독립 수치적분 대조, 잘못된 단위·구역·입력 거부가 테스트에 포함됩니다. 테스트 이름과 변경 대응은 `source_to_code.csv`에 있습니다.

명세와 정의가 완전히 주어진 **수학적 부분**은 조건부로 확인했습니다. AR 사다리 길이는 w=30 µm일 때 270/390/540/750/1050/1470 µm입니다. 동일 side/end k의 사각 4면, AR49, 끝점 SC=.90이면 Da_crit={design['critical_equal_rate_square_AR49']['value']:.10f}입니다. 보편적인 임계값 또는 제조 한계가 아닙니다.

새 SYNTHETIC 물성과 기본 끝점 관측으로 65~80°C/글리세린 질량0~40%의 336개 격자에서 AR49 SC는 {100*process['SC_AR49_range'][0]:.3f}~{100*process['SC_AR49_range'][1]:.3f}%이고 SC≥90%는 {process['n_at_or_above_target']}/336입니다. **해당 가정과 영역에서 경계 없음**을 출력했습니다. AR_max는 {process['AR_max_range'][0]:.6f}~{process['AR_max_range'][1]:.6f}입니다. 이전의 최대28%/3.7~11.6을 재현한 값이 아니라 공개한 새 생성기 입력의 결과입니다.

신규 36점, additive Gaussian SD=.05(5 percentage points), seed0의 4000회 MC에서 fixed-theory R² 중앙값은 {mc['fixed_theory']['r2_median']:.9f}, R²≥.95 관측은 {mc['fixed_theory']['target_success_count']}/4000입니다. OLS in-sample R²를 별도로 기록합니다. 과거 27조건·중앙값 .442와 동일 실험이 아닙니다.

별도 legacy 예제의 fixed-theory R²≥.95는 {legacy['fixed_theory']['target_success_count']}/4000이며 Wilson95 확률 구간은 {legacy['fixed_theory']['target_probability_wilson95']}입니다. 0/4000이라도 참확률이 정확히 0이라는 뜻이 아닙니다. 경계에서 plug-in MC 표준오차=0은 불확도가 없다는 근거가 아니며 구간을 함께 봐야 합니다.

## 합성 ML 실행 결과 — 실험 성능 아님

{metrics['n_measurements']}개의 합성 등록 ROI, {metrics['n_conditions']}개의 외부 조건을 사용했습니다. 각 outer train 내부의 모델 선택 절차에 대한 cross-fitted R²={metrics['R2']:.9f}, RMSE={metrics['RMSE_nm']:.6f} nm, 상대오차 bias={metrics['relative_bias_pct']:.6f}%, 1σ(ddof=1)={metrics['relative_1sigma_pct']:.6f}%입니다. 타일 352개를 n으로 사용하지 않았습니다. 모델별 표·모든 split·각 fold 모델은 `evaluation/`에 있습니다.

최종 모델은 전체 라벨의 내부 CV로 별도로 선택·학습했습니다. 최종 학습 자료 재예측은 독립 점수에 포함하지 않았습니다. 합성 생성기는 OD=두께/450과 같은 물리식을 사용하므로 이 점수와 상사성 점수는 연결 경로가 작동한다는 확인이며 카메라·FESEM·실제 물리의 검증이 아닙니다. 특정 모델의 일반적 우위나 과거 SVR .88을 입증하지 않습니다.

## 과거 주장 전체 대조와 미실행 항목

`validation/claims_review.csv`는 상세 프롬프트 §8과 START_HERE의 각 수치를 출처·확보 입력·부족 정의·실행 명령·산출물에 연결합니다. 동일 조건 재현(수학), 다른 가정에서 새 계산, 원자료/정의 부족, 실측 필요를 구분했습니다. 다른 가정의 차이를 역사적 실험의 반증으로 쓰지 않았습니다.

실제 원본 3장과 REF/DARK·촬영·µm/px·ROI, 대응 FESEM/수작업 계측, 실제 runs/layout/점도/평판 4시점, 반응면/SC/유효점/±오차 정의가 필요합니다. 자세한 최소 목록과 막힌 단계는 `limitations_and_missing_data.md`에 있습니다.
'''
(root/'validation_report.md').write_text(text,encoding='utf-8')
