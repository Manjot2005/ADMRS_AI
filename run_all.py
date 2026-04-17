"""
run_all.py — Run the complete ForestWatch AI pipeline end-to-end.
Stages 2-5 work in demo mode without real satellite data.
Stage 1 requires Copernicus credentials.

Usage:
    python run_all.py            # Runs stages 2, 3, 4, 5 (demo mode)
    python run_all.py --all      # Runs all 5 stages including download
"""

import sys
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def run_stage(name: str, module_path: str):
    log.info(f"\n{'='*60}")
    log.info(f"▶ Running {name}")
    log.info(f"{'='*60}")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("stage", module_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        log.info(f"✅ {name} completed successfully\n")
        return True
    except Exception as e:
        log.error(f"❌ {name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="ForestWatch AI Pipeline Runner")
    parser.add_argument("--all", action="store_true", help="Include stage 1 (satellite download)")
    parser.add_argument("--stage", type=int, help="Run only a specific stage (1-5)")
    args = parser.parse_args()

    stages = [
        (1, "Stage 1: Data Acquisition",       "stage1_download.py"),
        (2, "Stage 2: Pre-processing",          "stage2_preprocess.py"),
        (3, "Stage 3: Model Training",          "stage3_train_model.py"),
        (4, "Stage 4: Inference",               "stage4_inference.py"),
        (5, "Stage 5: Change Detection & Alert","stage5_change_detect.py"),
    ]

    if args.stage:
        stages = [s for s in stages if s[0] == args.stage]
        if not stages:
            log.error(f"Invalid stage: {args.stage}. Must be 1-5.")
            sys.exit(1)
    elif not args.all:
        stages = stages[1:]  # Skip stage 1 by default
        log.info("Skipping Stage 1 (use --all to include satellite download)")

    results = {}
    for num, name, path in stages:
        results[name] = run_stage(name, path)

    log.info("\n" + "=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)
    for name, ok in results.items():
        status = "✅" if ok else "❌"
        log.info(f"  {status} {name}")

    log.info("\n🚀 Launch dashboard with:")
    log.info("   streamlit run dashboard/app.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
