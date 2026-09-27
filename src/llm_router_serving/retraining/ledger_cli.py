"""Operator CLI for reconciling the retraining ledger with a manual CURRENT rollback.

`model_publish_cli rollback` (ml/) restores the GCS CURRENT pointer only. The
serving ledger (SQLite `retraining_jobs`) is a separate source of truth that
`build_continual_training_plan` uses to pick the next direct-promotion baseline;
see docs/ml/README.md 5.14 for the full procedure. Run this against the same
`retraining.sqlite3` the server uses, after a manual rollback and before the
next `/escalate` in direct mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_router_serving.retraining.store import (
    LedgerBatchNotFoundError,
    LedgerReconciliationError,
    RetrainingJobStore,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="Path to the server's retraining.sqlite3 (e.g. /app/var/r0/retraining.sqlite3)",
    )
    parser.add_argument(
        "--batch-id",
        required=True,
        help="batchId of the succeeded run whose CURRENT pointer was manually rolled back",
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="Short note for the server log, e.g. the rollback CLI invocation or ticket",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = RetrainingJobStore(args.db_path)
    try:
        job = store.reconcile_manual_rollback(batch_id=args.batch_id, reason=args.reason)
    except LedgerBatchNotFoundError:
        print(f"No ledger row for batch_id={args.batch_id!r}", file=sys.stderr)
        return 1
    except LedgerReconciliationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "batch_id": job.batch_id,
                "status": job.status,
                "logical_model_version": job.logical_model_version,
                "candidate_model_version": job.candidate_model_version,
                "promotion_mode": job.promotion_mode,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
