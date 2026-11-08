"""Build a ``POST /predict`` body from an FD001 test engine (thin CLI; docs/06 §7).

Examples::

    python scripts/make_request_payload.py --unit 1 > /tmp/payload.json
    python scripts/make_request_payload.py --unit 1 --last 30 \
        --out examples/predict_request_min.json --expected examples/expected_predictions.json
"""

import argparse
import json
import sys
from pathlib import Path

from predmaint.api.service import expected_prediction, request_payload
from predmaint.config import get_settings
from predmaint.data.load import load_fd001
from predmaint.models.predict import ModelNotAvailableError, load_interval_model, load_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unit", type=int, default=1, help="FD001 test engine id (1-100)")
    parser.add_argument("--last", type=int, default=None, help="keep only the last N cycles")
    parser.add_argument("--out", type=Path, default=None, help="write here instead of stdout")
    parser.add_argument(
        "--expected", type=Path, default=None, help="update this sidecar with the expected result"
    )
    args = parser.parse_args()

    settings = get_settings()
    data = load_fd001(settings.cmapss_dir)
    engine = data.test[data.test["unit"] == args.unit]
    if engine.empty:
        print(f"ERROR: no test engine {args.unit}", file=sys.stderr)
        return 1
    if args.last is not None:
        engine = engine.tail(args.last)
    payload = request_payload(engine, unit_id=f"test-unit-{args.unit}")

    if args.out is None:
        print(json.dumps(payload))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.expected is not None:
        sidecar = json.loads(args.expected.read_text()) if args.expected.is_file() else {}
        true_rul = int(data.rul["rul"].iloc[args.unit - 1])
        try:
            intervals = load_interval_model()
        except ModelNotAvailableError:
            intervals = None
        sidecar[args.out.name] = expected_prediction(payload, load_model(), true_rul, intervals)
        args.expected.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
