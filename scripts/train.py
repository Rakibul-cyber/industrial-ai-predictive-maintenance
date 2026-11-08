"""Train and evaluate an FD001 RUL model (thin CLI over ``predmaint.models.train``).

Usage::

    python scripts/train.py --model baseline
    python scripts/train.py --model primary
    python scripts/train.py --select
    python scripts/train.py --model intervals   # after --select (ADR-034)
    python scripts/train.py --model anomaly     # after --select (ADR-035)
"""

from predmaint.models.train import main

if __name__ == "__main__":
    raise SystemExit(main())
