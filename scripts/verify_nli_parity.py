#!/usr/bin/env python3
"""Replay the offline NLI evaluation pairs and compare against the stored scores.

The NLI cache threshold (0.28154424) was measured with
``sentence_transformers.CrossEncoder.predict(apply_softmax=True)`` on pairs reduced
by the experiment's ``diffpair.reduce_pair``. Serving reproduces both with plain
``transformers`` and ``sentence_diff.reduce_pair``. This checks that they agree:
if a pair changes side of the threshold, the threshold is not valid here.

Input is JSON Lines with ``query``, ``candidate`` (the original, unreduced texts)
and ``expected`` (the offline bidirectional-entailment minimum).

    python scripts/verify_nli_parity.py \
        --jsonl /path/to/nli-parity.jsonl \
        --model /opt/models/deberta-base-long-nli
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_router_serving.inference import TransformersNliReranker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=1280)
    # Batching/padding noise is around 1e-6; a wrong column or activation moves
    # scores by 0.1 or more.
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.28154424)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    args = parser.parse_args()

    with args.jsonl.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        print("JSONL에 행이 없습니다", file=sys.stderr)
        return 2

    engine = TransformersNliReranker.load(
        model_id=args.model,
        model_version="parity-check",
        max_length=args.max_length,
        device_name=args.device,
        local_files_only=args.local_files_only,
    )

    worst = 0.0
    numeric_failures = 0
    decision_flips = 0
    for number, row in enumerate(rows):
        expected = float(row["expected"])
        # One candidate per call so batching cannot mask a per-pair drift.
        actual = engine.rerank(row["query"], [row["candidate"]]).scores[0]
        delta = abs(actual - expected)
        worst = max(worst, delta)
        if delta > args.tolerance:
            numeric_failures += 1
            print(f"수치 불일치 #{number}: 기대 {expected:.8f} 실제 {actual:.8f} 차이 {delta:.2e}")
        if (expected >= args.threshold) != (actual >= args.threshold):
            decision_flips += 1
            print(
                f"판정 뒤집힘 #{number}: 기대 {expected:.8f} 실제 {actual:.8f} "
                f"(임계값 {args.threshold})"
            )

    print(f"\n{len(rows)}쌍 검증")
    print(f"  최대 점수 차이 : {worst:.2e}  (허용 {args.tolerance:.0e})")
    print(f"  허용 초과      : {numeric_failures}건")
    print(f"  판정 뒤집힘    : {decision_flips}건  (임계값 {args.threshold})")

    if decision_flips:
        print("\n실패 — 임계값 기준 판정이 달라진다. 이 구현으로 임계값을 재사용하면 안 된다.")
        return 1
    if numeric_failures:
        print("\n경고 — 판정은 같지만 점수가 허용 오차를 넘는다. 원인을 확인한다.")
        return 1
    print("\n통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
