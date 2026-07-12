"""Tiny policy retrieval layer backing the agent's `search_policy` tool.

Deliberately simple keyword scoring over heading-level sections — the point
of this project is the escalation architecture, not the retriever. Swapping
this for embeddings changes nothing about the agent contract: the tool still
returns (document, section, text) triples the model must cite.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

_WORD = re.compile(r"[a-z]{3,}")

_STOPWORDS = {
    "the", "and", "for", "you", "your", "are", "can", "how", "what", "when",
    "does", "with", "this", "that", "have", "has", "will", "policy", "about",
}


@dataclass
class PolicySection:
    document: str
    section: str
    text: str


class PolicyCorpus:
    def __init__(self, sections: List[PolicySection]):
        self.sections = sections

    @classmethod
    def load(cls, policies_dir: Path) -> "PolicyCorpus":
        sections: List[PolicySection] = []
        for path in sorted(policies_dir.glob("*.md")):
            sections.extend(_split_sections(path))
        return cls(sections)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        terms = [t for t in _WORD.findall(query.lower()) if t not in _STOPWORDS]
        if not terms:
            return []
        scored = []
        for sec in self.sections:
            haystack = f"{sec.section} {sec.text}".lower()
            # section-title hits weigh 3x body hits
            score = sum(
                3 * sec.section.lower().count(t) + haystack.count(t)
                for t in terms
            )
            if score > 0:
                scored.append((score, sec))
        scored.sort(key=lambda pair: -pair[0])
        return [
            {"document": sec.document, "section": sec.section, "text": sec.text}
            for _, sec in scored[:top_k]
        ]


def _split_sections(path: Path) -> List[PolicySection]:
    """Split a markdown doc into one PolicySection per `## heading`."""
    doc = path.stem
    sections: List[PolicySection] = []
    current_heading = "Preamble"
    current_lines: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if current_lines and any(l.strip() for l in current_lines):
                sections.append(PolicySection(doc, current_heading, "\n".join(current_lines).strip()))
            current_heading = line[3:].strip()
            current_lines = []
        elif line.startswith("# "):
            continue  # doc title
        else:
            current_lines.append(line)
    if current_lines and any(l.strip() for l in current_lines):
        sections.append(PolicySection(doc, current_heading, "\n".join(current_lines).strip()))
    return sections
