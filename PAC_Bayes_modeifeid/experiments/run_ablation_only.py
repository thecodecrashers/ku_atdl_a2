"""Run all three concurrent ablations with the fast runner options."""

import os
import sys

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_ROOT)

from experiments.run_ablation_fast import main


if __name__ == "__main__":
    # Keep the old entry-point name while routing to the single implementation
    # of the three conditions. This avoids divergent ablation definitions.
    raise SystemExit(main())
