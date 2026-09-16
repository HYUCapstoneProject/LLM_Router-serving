#!/usr/bin/env python3
"""Replay the offline screening pairs and compare against the stored scores.

The candidate cache threshold (0.03581769) was measured with
``sentence_transformers.CrossEncoder.predict``. Serving reproduces that scoring
with plain ``transformers`` plus a sigmoid, because the current
sentence-transformers release requires transformers>=5 and would drag the BGE
embedder and the ModernBERT classifier along with it.

A version pin would only express the hope that the two agree. This checks it:
if the scores drift, the threshold is no longer valid and the run fails.

    python scripts/verify_reranker_parity.py \
        --csv /path/to/cross-encoder--quora-distilroberta-base.csv \
        --model /opt/models/quora-distilroberta-base
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_router_serving.inference import TransformersCrossEncoderReranker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=512)
    # float32 noise from different batching/padding shows up around 1e-6.
    # An activation mismatch would move scores by 0.1 or more, so this is loose
    # enough to ignore rounding and still four orders tighter than that.
    parser.add_argument("--tolerance", type=float, default=1e-4)
    # What actually has to hold: no pair changes side of the candidate threshold.
    parser.add_argument("--threshold", type=float, default=0.03581769)
    parser.add_argument("--local-files-only", action="store_true", default=False)
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("CSV에 행이 없습니다", file=sys.stderr)
        return 2

    engine = TransformersCrossEncoderReranker.load(
        model_id=args.model,
        model_version="parity-check",
        max_length=args.max_length,
        device_name=args.device,
        local_files_only=args.local_files_only,
    )

    worst = 0.0
    numeric_failures = 0
    decision_flips = 0
    for row in rows:
        expected = float(row["reranker_score"])
        # One pair per call so batching and padding cannot mask a per-pair drift.
        actual = engine.rerank(row["query"], [row["candidate"]]).scores[0]
        delta = abs(actual - expected)
        worst = max(worst, delta)

        if delta > args.tolerance:
            numeric_failures += 1
            print(
                f"수치 불일치 {row['pair_id']}: "
                f"기대 {expected:.8f} 실제 {actual:.8f} 차이 {delta:.2e}"
            )

        # This is what the cache actually depends on. A score can drift by float32
        # noise without consequence; a pair changing side of the threshold cannot.
        if (expected >= args.threshold) != (actual >= args.threshold):
            decision_flips += 1
            print(
                f"판정 뒤집힘 {row['pair_id']}: "
                f"기대 {expected:.8f} 실제 {actual:.8f} (임계값 {args.threshold})"
            )

    print(f"\n{len(rows)}쌍 검증")
    print(f"  최대 점수 차이 : {worst:.2e}  (허용 {args.tolerance:.0e})")
    print(f"  판정 뒤집힘    : {decision_flips}건  (임계값 {args.threshold})")

    if decision_flips:
        print("\n실패 — 임계값 기준 판정이 달라진다. 이 구현으로 임계값을 재사용하면 안 된다.")
        return 1
    if numeric_failures:
        print("\n실패 — 점수 차이가 허용치를 넘는다. 활성함수가 다를 수 있다.")
        return 1
    print("\n일치. 스크리닝에서 측정한 임계값을 서빙에 그대로 쓸 수 있다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
