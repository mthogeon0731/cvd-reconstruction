import argparse
import json
from pathlib import Path
import pandas as pd
from .config import Config
from .demo import run_demo, write_json
from .data import evaluate_conditions
from .monte_carlo import noise_stress_test
from .io_utils import read_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description="CVD/CBD code reconstruction scaffold (uncalibrated)")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run synthetic simulator + optional proposal RF + noise stress test")
    demo.add_argument("--config", default="configs/reference_demo.json")
    demo.add_argument("--out", default="outputs/demo")
    demo.add_argument("--no-plots", action="store_true")
    demo.add_argument("--mc-repeats", type=int, default=4000)
    simulate = sub.add_parser("simulate", help="Calculate SC proxies from a physical-condition CSV")
    simulate.add_argument("--config", default="configs/reference_demo.json")
    simulate.add_argument("--input", required=True)
    simulate.add_argument("--output", required=True)
    noise = sub.add_parser("monte-carlo", help="Noise stress test on a CSV with sc_proxy column")
    noise.add_argument("--input", required=True)
    noise.add_argument("--out", required=True)
    noise.add_argument("--repeats", type=int, default=4000)
    noise.add_argument("--noise-sd", type=float, default=.05)
    noise.add_argument("--noise-kind", choices=["additive", "relative"], default="additive")
    noise.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = run_demo(Config.load(args.config), args.out, not args.no_plots, args.mc_repeats)
            print(json.dumps({"output": str(Path(args.out).resolve()), "status": result["status"],
                              "samples": result["data_count"], "runs": result["doe_selected_count"]}, indent=2))
        elif args.command == "simulate":
            target = Path(args.output)
            if target.exists():
                raise ValueError("Output already exists; choose another path")
            result = evaluate_conditions(read_csv(args.input), Config.load(args.config))
            target.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(target, index=False)
            print(f"Saved {len(result)} synthetic condition predictions to {target}")
        elif args.command == "monte-carlo":
            out = Path(args.out)
            if out.exists() and any(out.iterdir()):
                raise ValueError("Output directory is not empty")
            trials, summary = noise_stress_test(read_csv(args.input)["sc_proxy"].to_numpy(),
                                               args.repeats, args.noise_sd, args.seed,
                                               noise_kind=args.noise_kind)
            out.mkdir(parents=True, exist_ok=True)
            trials.to_csv(out / "trials.csv", index=False)
            write_json(out / "summary.json", summary)
            print(json.dumps(summary, indent=2))
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(2, f"Input error: {exc}\n")
