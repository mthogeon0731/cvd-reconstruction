# Fourteen study CLI stages

From the repository root after installation, first generate the inputs:

```sh
python -m cvd_cbd study generate --config configs/synthetic_study.json --root outputs/generated_inputs
```

Then execute these 14 commands in order. Use a fresh `outputs/replay` directory. The model loaded below was just fitted locally; never trust an unrelated joblib file.
```bash
# 설정·자료/조인 검사
python -m cvd_cbd study validate --config configs/synthetic_study.json --root outputs/generated_inputs

# 01 독립 평판 캘리브레이션
python -m cvd_cbd study calibrate --config configs/synthetic_study.json --root outputs/generated_inputs --out outputs/replay/01

# 00 설계·관측량·민감도·Monte Carlo
python -m cvd_cbd study design --config configs/synthetic_study.json --calibration outputs/replay/01/calibration.csv --out outputs/replay/00

# 02 manifest: 파일명, REF/DARK, 중복 픽셀, 조인과 단위 검증
python -m cvd_cbd study manifest --config configs/synthetic_study.json --root outputs/generated_inputs --out outputs/replay/02

# 03 TIFF → T/OD → 경계/ROI → 타일 → ROI 특징
python -m cvd_cbd study extract --config configs/synthetic_study.json --root outputs/generated_inputs --manifest outputs/replay/02/manifest.csv --out outputs/replay/03

# 04 외부 조건 LOGO와 내부 조건 CV, 모델 종류까지 내부 선택
python -m cvd_cbd study compare --config configs/synthetic_study.json --features outputs/replay/03/features.csv --manual outputs/generated_inputs/meta/manual_om.csv --out outputs/replay/04

# 05 전체 라벨로 별도 최종 학습, 추론 및 SC
python -m cvd_cbd study fit-final --config configs/synthetic_study.json --features outputs/replay/03/features.csv --out outputs/replay/final_model
python -m cvd_cbd study predict --config configs/synthetic_study.json --features outputs/replay/03/features.csv --model outputs/replay/final_model/final_model.joblib --output outputs/replay/predictions_final.csv --trust-model
python -m cvd_cbd study sc --config configs/synthetic_study.json --features outputs/replay/predictions_final.csv --out outputs/replay/sc_final

# 06 검증에는 cross-fitted 예측 사용; 최종 모델 재예측은 거부
python -m cvd_cbd study sc --config configs/synthetic_study.json --features outputs/replay/04/selected_oof.csv --out outputs/replay/sc_oof
python -m cvd_cbd study similitude --config configs/synthetic_study.json --trenches outputs/replay/sc_oof/trench_sc.csv --calibration outputs/replay/01/calibration.csv --out outputs/replay/06

# FE-SEM 직접 SC는 별도 집계
python -m cvd_cbd study sc --config configs/synthetic_study.json --features outputs/replay/03/features.csv --source fesem --out outputs/replay/sc_fesem

# 07 조성 효과를 유지한 공정영역/AR_max
python -m cvd_cbd study process-window --config configs/synthetic_study.json --calibration outputs/replay/01/calibration.csv --out outputs/replay/07

# 물성/기하/두께의 결합 불확도 전파
python -m cvd_cbd study uncertainty --config configs/synthetic_study.json --input outputs/generated_inputs/uncertainty.json --out outputs/replay/uncertainty
```
`study run --config configs/synthetic_study.json --root outputs/generated_inputs --out outputs/run` orchestrates these operations. `study demo` creates new inputs and orchestrates them. All results are synthetic software checks.

The separate `scripts/validate_cli.py` smoke tool also calls 14 commands but includes rejection checks and is not the same procedure. See [validation](VALIDATION.md) for what was actually executed and [input contracts](INPUT_CONTRACTS.md) for rejection rules.