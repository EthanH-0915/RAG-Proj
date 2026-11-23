"""Answer a question from retrieved chunks, with a source ID cited for every claim.

    retrieve -> decline check -> prompt -> local LLM -> citation check

Declining happens in code, before the model is called, on the two signals the
retrieval evaluation found: the question names an ID that matches no chunk
("What is T9999?"), or nothing in the corpus is close to it (top vector
similarity below MIN_SIMILARITY). A 3B model is not reliable at noticing that
its sources don't answer, though it is still told to decline, as a backstop.

The model answers in a JSON schema, one entry per sentence with the IDs it rests
on. The schema allows only the IDs of the sources actually sent, and at least
one per sentence, so Ollama's constrained decoding rules out invented
citations; the same rules are checked again in code in case a runtime ignores
the schema. A failing reply gets one retry with its problems listed; sentences
that still fail are dropped and the answer is flagged.

Usage:
    python -m rag.answer "Which mitigations help against password spraying?"
    python -m rag.answer "What is T1059.001?" --mode hybrid-pin -k 3 --show-sources
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field

from . import config, db, hybrid, llm
from .query import RETRIEVERS, search as vector_search

# Bump when SYSTEM_PROMPT, the source format or the trimming changes, so cached
# or saved answers from an older prompt are never compared with new ones.
PROMPT_VERSION = "1"
DEFAULT_MODE = "hybrid+expand"
DEFAULT_K = 5

# Top cosine similarity below which a question counts as off-topic. It sits
# between the lowest answerable question in eval/questions.yaml (0.36) and the
# one off-topic question (banana bread, 0.15): a single point on the low side,
# so revisit it as more off-topic questions are added.
MIN_SIMILARITY = 0.25

# Character budgets, at roughly 4 characters per token. Technique chunks run to
# 13,000 characters; a handful of those would crowd a 3B model's attention
# even where they fit its window.
SOURCE_CHARS = 2400
TOTAL_CHARS = 12000

SYSTEM_PROMPT = """\
You answer questions about cyber threats using only the sources in the user's \
message: MITRE ATT&CK techniques, mitigations and threat groups, and CISA's \
Known Exploited Vulnerabilities catalog. Each source starts with its ID in \
square brackets, such as [T1110.003], [M1032], [G0016] or [CVE-2021-44228].

Reply in JSON. Put each sentence of your answer in "answer", with the IDs of \
the sources that support that sentence in "sources".
- Use only facts stated in the sources. Do not add outside knowledge.
- Every sentence needs at least one source ID, and only IDs of sources you \
were given.
- Answer in one to five sentences, most important first.
- If the sources do not answer the question, reply with an empty "answer" and \
"declined": true."""

# Bracketed IDs the model sometimes also writes into the sentence text; the
# rendered answer adds its own, so these are stripped.
_INLINE_CITATION = re.compile(r"\s*\[(?:[A-Z]{1,3}[\d.\-]+(?:,\s*)?)+\]")


@dataclass
class Answer:
    question: str
    sentences: list[dict]  # [{"text": ..., "sources": [...]}]
    declined: bool
    reason: str | None = None  # why it declined, or why it is flagged
    flagged: bool = False  # some sentences failed the citation check twice
    mode: str = DEFAULT_MODE
    retrieved: list[str] = field(default_factory=list)
    sources_sent: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    attempts: int = 0
    timings: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)
    model: str = config.LLM_MODEL
    prompt_version: str = PROMPT_VERSION

    def text(self) -> str:
        if self.declined:
            return f"I can't answer that from the sources: {self.reason}"
        return " ".join(f"{s['text']} [{', '.join(s['sources'])}]" for s in self.sentences)


def answer(question: str, mode: str = DEFAULT_MODE, k: int = DEFAULT_K) -> Answer:
    result = Answer(question=question, sentences=[], declined=False, mode=mode)
    start = time.perf_counter()

    unknown = _unknown_ids(hybrid.extract_ids(question))
    if unknown:
        result.declined = True
        result.reason = f"no source in the corpus has the ID {', '.join(unknown)}."
        result.timings["retrieval_s"] = time.perf_counter() - start
        return result

    top = vector_search(question, 1)
    similarity = float(top[0]["similarity"]) if top else 0.0
    if similarity < MIN_SIMILARITY:
        result.declined = True
        result.reason = f"nothing in the corpus is close to this question (top similarity {similarity:.2f})."
        result.timings["retrieval_s"] = time.perf_counter() - start
        return result

    hits = RETRIEVERS[mode](question, k)
    result.retrieved = [h["source_id"] for h in hits]
    sources = _sources(hits)
    result.sources_sent = [sid for sid, _ in sources]
    result.timings["retrieval_s"] = time.perf_counter() - start
    if not sources:
        result.declined = True
        result.reason = "retrieval returned no sources."
        return result

    allowed = set(result.sources_sent)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_message(question, sources)},
    ]
    schema = _schema(result.sources_sent)
    llm_seconds = prompt_tokens = output_tokens = 0
    valid, problems = [], []
    for attempt in (1, 2):
        reply = llm.chat(messages, schema)
        result.attempts = attempt
        llm_seconds += reply.seconds
        prompt_tokens += reply.prompt_tokens
        output_tokens += reply.output_tokens
        parsed, valid, problems = _check(reply.content, allowed)
        if not problems:
            break
        messages += [
            {"role": "assistant", "content": reply.content},
            {
                "role": "user",
                "content": "Your answer broke the rules:\n- "
                + "\n- ".join(problems)
                + "\nAnswer again, following every rule.",
            },
        ]
    result.timings["llm_s"] = llm_seconds
    result.tokens = {"prompt": prompt_tokens, "output": output_tokens}
    result.problems = problems

    if valid:
        result.sentences = valid
        if problems:
            result.flagged = True
            result.reason = f"{len(problems)} sentence(s) dropped for failing the citation check."
    elif parsed.get("declined") and not problems:
        result.declined = True
        result.reason = "the retrieved sources do not answer it (the model declined)."
    else:
        result.declined = True
        result.flagged = bool(problems)
        result.reason = "the model gave no sentence that passed the citation check."
    return result


def _unknown_ids(ids: list[str]) -> list[str]:
    if not ids:
        return []
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT source_id FROM chunks WHERE embedding_model = %s AND source_id = ANY(%s)",
            (config.EMBEDDING_MODEL, ids),
        )
        known = {row[0] for row in cur.fetchall()}
    return [i for i in ids if i not in known]


def _sources(hits: list[dict]) -> list[tuple[str, str]]:
    """(source_id, trimmed content) in retrieval order, strongest first, until
    TOTAL_CHARS is used up."""
    out, used = [], 0
    for hit in hits:
        content = _trim(hit["content"], SOURCE_CHARS)
        if used + len(content) > TOTAL_CHARS:
            break
        out.append((hit["source_id"], content))
        used += len(content)
    return out


def _trim(content: str, limit: int) -> str:
    """Fit a chunk into `limit` characters. The Detection section goes first:
    it is the longest part of most technique chunks and rarely what a question
    asks about. Then the description is shortened, so the relationship
    sections after it (mitigation guidance, groups, CVEs) survive, and only
    then is the tail cut."""
    if len(content) <= limit:
        return content
    sections = [s for s in content.split("\n\n") if not s.startswith("Detection:")]
    excess = len("\n\n".join(sections)) - limit
    if excess > 0:
        for i, section in enumerate(sections):
            if section.startswith("Description:"):
                keep = max(200, len(section) - excess - len(" [...]"))
                sections[i] = section[:keep].rstrip() + " [...]"
                break
    out = "\n\n".join(sections)
    if len(out) > limit:
        out = out[:limit].rsplit("\n", 1)[0] + "\n[...]"
    return out


def _user_message(question: str, sources: list[tuple[str, str]]) -> str:
    # Sources first and the question last: small models attend best to the
    # end of a long prompt.
    blocks = "\n\n".join(f"[{sid}]\n{content}" for sid, content in sources)
    return f"Sources:\n\n{blocks}\n\nQuestion: {question}"


def _schema(source_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "answer": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "sources": {
                            "type": "array",
                            "items": {"type": "string", "enum": source_ids},
                            "minItems": 1,
                        },
                    },
                    "required": ["text", "sources"],
                },
            },
            "declined": {"type": "boolean"},
        },
        "required": ["answer", "declined"],
    }


def _check(content: str, allowed: set[str]) -> tuple[dict, list[dict], list[str]]:
    """(parsed reply, sentences that pass, problems) for one model reply."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {}, [], ["The reply was not valid JSON."]
    valid, problems = [], []
    for n, sentence in enumerate(parsed.get("answer") or [], start=1):
        text = _INLINE_CITATION.sub("", str(sentence.get("text", ""))).strip()
        if not text:
            continue
        cited = list(dict.fromkeys(str(c).strip(" []").upper() for c in sentence.get("sources") or []))
        bad = [c for c in cited if c not in allowed]
        if not cited:
            problems.append(f'Sentence {n} cites no source: "{text}"')
        elif bad:
            problems.append(f'Sentence {n} cites {", ".join(bad)}, which is not one of the sources: "{text}"')
        else:
            valid.append({"text": text, "sources": cited})
    return parsed, valid, problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--mode", choices=list(RETRIEVERS), default=DEFAULT_MODE)
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"chunks to retrieve (default {DEFAULT_K})")
    parser.add_argument("--show-sources", action="store_true", help="list the sources sent to the model")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args()

    try:
        result = answer(args.question, args.mode, args.k)
    except llm.LLMError as e:
        raise SystemExit(str(e)) from e

    if args.json:
        print(json.dumps(asdict(result), indent=2))
        return
    print(f"\nQ: {args.question}\n")
    print(result.text())
    if result.flagged:
        print(f"\n(flagged: {result.reason})")
    if args.show_sources and result.sources_sent:
        print("\nSources sent: " + ", ".join(result.sources_sent))
    t = result.timings
    line = f"\nretrieval {t.get('retrieval_s', 0):.1f}s"
    if "llm_s" in t:
        line += (
            f", {result.model} {t['llm_s']:.1f}s ({result.attempts} attempt(s), "
            f"{result.tokens['prompt']} prompt + {result.tokens['output']} output tokens)"
        )
    print(line)


if __name__ == "__main__":
    main()
