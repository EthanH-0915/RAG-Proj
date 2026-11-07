"""Score retrieval against the hand-written questions in eval/questions.yaml.

Each question whose `needs` doc_types are ingested is run through retrieval, and
the returned source IDs are compared with its `expected_source_ids`:

    recall@k  share of the expected IDs found in the top k
    MRR@k     mean of 1 / rank of the first expected ID (0 if none is in the top k)

`should_decline` questions have nothing correct to retrieve, so they stay out of
both averages; their top similarity is reported instead, since that is what a
"no good match" threshold for the generation step will be set from.

Usage:
    python -m rag.evaluate
    python -m rag.evaluate -k 10 --out eval/results/vector-k10.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import yaml

from . import config, db, query

QUESTIONS_PATH = config.ROOT / "eval" / "questions.yaml"

# retrieval_mode -> search(question, k), returning ranked hits with `source_id`
# and `similarity`. Hybrid retrieval joins here in week 4.
RETRIEVERS = {"vector": query.search}


def load_questions(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def corpus_state() -> tuple[set[str], set[str], dict]:
    """Doc types and source IDs ingested with the current model, and the latest manifest."""
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT doc_type, source_id FROM chunks WHERE embedding_model = %s",
            (config.EMBEDDING_MODEL,),
        )
        rows = cur.fetchall()
        cur.execute(
            """SELECT manifest FROM ingest_runs WHERE embedding_model = %s
               ORDER BY finished_at DESC LIMIT 1""",
            (config.EMBEDDING_MODEL,),
        )
        run = cur.fetchone()
    return {r[0] for r in rows}, {r[1] for r in rows}, (run[0] if run else {})


def score(expected: list[str], retrieved: list[str]) -> tuple[float, float]:
    """(recall, reciprocal rank) of one ranked list against the expected IDs."""
    recall = len(set(expected) & set(retrieved)) / len(expected)
    rr = next(
        (1 / rank for rank, sid in enumerate(retrieved, start=1) if sid in expected),
        0.0,
    )
    return recall, rr


def evaluate(
    questions: list[dict], mode: str, k: int, doc_types: set[str], source_ids: set[str]
) -> list[dict]:
    search = RETRIEVERS[mode]
    results = []
    for q in questions:
        result = {"id": q["id"], "category": q.get("category"), "question": q["question"]}
        expected = q.get("expected_source_ids") or []

        missing_types = sorted(set(q.get("needs", [])) - doc_types)
        if missing_types:
            results.append({**result, "status": "skipped", "reason": f"needs {', '.join(missing_types)}"})
            continue
        if not expected and not q.get("should_decline"):
            results.append({**result, "status": "skipped", "reason": "no expected_source_ids"})
            continue

        hits = search(q["question"], k)
        retrieved = [h["source_id"] for h in hits]
        result["retrieved"] = retrieved
        result["top_similarity"] = float(hits[0]["similarity"]) if hits else None

        if q.get("should_decline"):
            results.append({**result, "status": "decline"})
            continue

        recall, rr = score(expected, retrieved)
        result |= {
            "status": "scored",
            "expected": expected,
            "recall": recall,
            "rr": rr,
            # Expected IDs absent from the corpus usually mean ATT&CK renumbered
            # something; they count as misses but are worth fixing in the YAML.
            "unknown_expected": sorted(set(expected) - source_ids),
        }
        results.append(result)
    return results


def summarise(results: list[dict]) -> dict:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r["status"] == "scored":
            by_category[r["category"]].append(r)
    scored = [r for rs in by_category.values() for r in rs]

    def agg(rs: list[dict]) -> dict:
        return {"n": len(rs), "recall": mean(r["recall"] for r in rs), "mrr": mean(r["rr"] for r in rs)}

    summary = {cat: agg(rs) for cat, rs in sorted(by_category.items())}
    if scored:
        summary["overall"] = agg(scored)
    return summary


def report(results: list[dict], summary: dict, k: int, header: str) -> None:
    print(f"\n{header}\n")
    print(f"{'id':<18} {'category':<14} {'R@' + str(k):>5} {'RR':>5} {'top sim':>8}  retrieved")
    for r in results:
        lead = f"{r['id']:<18} {r['category'] or '':<14}"
        if r["status"] == "skipped":
            print(f"{lead} skipped ({r['reason']})")
            continue
        sim = f"{r['top_similarity']:.3f}" if r["top_similarity"] is not None else "-"
        retrieved = ", ".join(r["retrieved"])
        if r["status"] == "decline":
            print(f"{lead} {'decline':>11} {sim:>8}  {retrieved}")
            continue
        print(f"{lead} {r['recall']:>5.2f} {r['rr']:>5.2f} {sim:>8}  {retrieved}")
        missed = [sid for sid in r["expected"] if sid not in r["retrieved"]]
        if missed:
            print(f"{'':<34}missed: {', '.join(missed)}")
        if r["unknown_expected"]:
            print(f"{'':<34}warning: not in corpus: {', '.join(r['unknown_expected'])}")

    print(f"\n{'category':<14} {'n':>3} {'recall@' + str(k):>9} {'MRR@' + str(k):>7}")
    for cat, s in summary.items():
        print(f"{cat:<14} {s['n']:>3} {s['recall']:>9.3f} {s['mrr']:>7.3f}")
    counts = defaultdict(int)
    for r in results:
        counts[r["status"]] += 1
    print(f"\n{counts['scored']} scored, {counts['decline']} decline, {counts['skipped']} skipped")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-k", type=int, default=5, help="chunks retrieved per question (default 5)")
    parser.add_argument("--mode", choices=sorted(RETRIEVERS), default="vector")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--out", type=Path, help="also write the results as JSON here")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    doc_types, source_ids, manifest = corpus_state()
    attack_version = manifest.get("attack", {}).get("version")
    kev_version = manifest.get("kev", {}).get("catalog_version")

    results = evaluate(questions, args.mode, args.k, doc_types, source_ids)
    summary = summarise(results)
    report(
        results,
        summary,
        args.k,
        f"mode={args.mode}  k={args.k}  model={config.EMBEDDING_MODEL}  "
        f"ATT&CK v{attack_version}  KEV {kev_version}  ingested: {', '.join(sorted(doc_types)) or 'none'}",
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": args.mode,
            "k": args.k,
            "embedding_model": config.EMBEDDING_MODEL,
            "attack_version": attack_version,
            "kev_catalog_version": kev_version,
            "doc_types": sorted(doc_types),
            "summary": summary,
            "questions": results,
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
