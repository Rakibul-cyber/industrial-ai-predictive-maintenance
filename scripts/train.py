"""Train and evaluate an FD001 RUL model (thin CLI over ``predmaint.models.train``).

Usage: ``python scripts/train.py --model baseline``
"""

from predmaint.models.train import main

if __name__ == "__main__":
    raise SystemExit(main())
