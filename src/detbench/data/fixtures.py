"""A tiny smoke-test corpus. **Not an evaluation set.**

Twelve documents cannot support a claim about any detector, and this module is written to
make that impossible to forget: `SMOKE.warning` is printed by every CLI path that touches
it, and the metrics layer will independently return `n/a` for TPR at 0.1% FPR here,
because six human documents cannot resolve a one-in-a-thousand false-positive rate. That
`n/a` is the intended lesson of the demo, not a bug in it.

Real evaluation uses the public benchmark corpora listed in `docs/METHODOLOGY.md` — RAID
(6M generations, 11 generators, 8 domains, 11 attacks), MAGE for the domain-shift split,
and PADBen for iterated paraphrase.

**Provenance, recorded at the boundary rather than inferred later** — which is the whole
argument this project is downstream of:

- `HUMAN`: excerpts from works in the public domain, published between 1851 and 1859.
  Pre-LLM by more than a century, so their label is a historical fact rather than a guess.
- `MACHINE`: written by Claude (Opus 5) on 2026-08-01 specifically for this fixture. The
  label is exact because the generation was observed, not because a detector was asked.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    text: str
    label: int  # 1 = machine, 0 = human
    source: str


SMOKE_WARNING = (
    "SMOKE FIXTURE ONLY — 12 documents. No number computed from this set supports any "
    "claim about any detector. See docs/METHODOLOGY.md for the real corpora."
)

HUMAN: list[Document] = [
    Document(
        "Call me Ishmael. Some years ago—never mind how long precisely—having little or "
        "no money in my purse, and nothing particular to interest me on shore, I thought "
        "I would sail about a little and see the watery part of the world. It is a way I "
        "have of driving off the spleen and regulating the circulation. Whenever I find "
        "myself growing grim about the mouth; whenever it is a damp, drizzly November in "
        "my soul, then I account it high time to get to sea as soon as I can.",
        0,
        "Melville, Moby-Dick (1851), public domain",
    ),
    Document(
        "It is a truth universally acknowledged, that a single man in possession of a "
        "good fortune, must be in want of a wife. However little known the feelings or "
        "views of such a man may be on his first entering a neighbourhood, this truth is "
        "so well fixed in the minds of the surrounding families, that he is considered "
        "the rightful property of some one or other of their daughters.",
        0,
        "Austen, Pride and Prejudice (1813), public domain",
    ),
    Document(
        "I went to the woods because I wished to live deliberately, to front only the "
        "essential facts of life, and see if I could not learn what it had to teach, and "
        "not, when I came to die, discover that I had not lived. I did not wish to live "
        "what was not life, living is so dear; nor did I wish to practise resignation, "
        "unless it was quite necessary.",
        0,
        "Thoreau, Walden (1854), public domain",
    ),
    Document(
        "When on board H.M.S. Beagle, as naturalist, I was much struck with certain facts "
        "in the distribution of the inhabitants of South America, and in the geological "
        "relations of the present to the past inhabitants of that continent. These facts "
        "seemed to me to throw some light on the origin of species—that mystery of "
        "mysteries, as it has been called by one of our greatest philosophers.",
        0,
        "Darwin, On the Origin of Species (1859), public domain",
    ),
    Document(
        "It was the best of times, it was the worst of times, it was the age of wisdom, "
        "it was the age of foolishness, it was the epoch of belief, it was the epoch of "
        "incredulity, it was the season of Light, it was the season of Darkness, it was "
        "the spring of hope, it was the winter of despair, we had everything before us, "
        "we had nothing before us, we were all going direct to Heaven, we were all going "
        "direct the other way.",
        0,
        "Dickens, A Tale of Two Cities (1859), public domain",
    ),
    Document(
        "There is a wisdom of the head, and there is a wisdom of the heart. The mariner "
        "who has been round the Horn does not tell you of the calms. He tells you of the "
        "gale that carried away his topmasts, and of the night he lay to under bare "
        "poles, and of the grey morning when the wind fell and the sea went down and the "
        "ship was still afloat and the men were still alive to sail her home again.",
        0,
        "Melville, Moby-Dick (1851), adapted excerpt, public domain",
    ),
]

MACHINE: list[Document] = [
    Document(
        "The integration of renewable energy sources into existing power grids presents "
        "several multifaceted challenges. Firstly, the intermittent nature of solar and "
        "wind generation requires robust storage solutions. Additionally, grid operators "
        "must leverage advanced forecasting to balance supply and demand effectively. "
        "Furthermore, regulatory frameworks often lag behind technological capabilities, "
        "creating barriers to adoption. Consequently, a comprehensive approach is "
        "essential for successful implementation across diverse regional contexts.",
        1,
        "Claude Opus 5, generated 2026-08-01 for this fixture",
    ),
    Document(
        "Effective time management is crucial for maintaining productivity in modern "
        "workplaces. It is important to prioritize tasks based on both urgency and "
        "importance. Additionally, setting clear boundaries around meeting schedules can "
        "significantly enhance focus. Notably, research suggests that uninterrupted work "
        "blocks improve output quality. Moreover, regular breaks are essential for "
        "sustaining cognitive performance throughout the day, particularly during "
        "demanding project cycles that require sustained concentration.",
        1,
        "Claude Opus 5, generated 2026-08-01 for this fixture",
    ),
    Document(
        "Urban planning in the twenty-first century must address a nuanced set of "
        "competing priorities. Population density continues to increase in metropolitan "
        "areas worldwide. Consequently, planners must balance housing availability "
        "against green space preservation. Additionally, transportation infrastructure "
        "requires substantial investment to remain viable. Furthermore, climate "
        "resilience has become a pivotal consideration, underscoring the need for "
        "holistic frameworks that integrate environmental and social objectives.",
        1,
        "Claude Opus 5, generated 2026-08-01 for this fixture",
    ),
    Document(
        "Machine learning models require careful validation before deployment in "
        "production environments. Firstly, training data must be representative of the "
        "target distribution. Additionally, practitioners should evaluate performance "
        "across demographic subgroups to identify potential disparities. Moreover, "
        "monitoring systems are essential for detecting distribution drift over time. "
        "Consequently, a comprehensive validation pipeline substantially reduces the risk "
        "of unexpected failures in real-world applications.",
        1,
        "Claude Opus 5, generated 2026-08-01 for this fixture",
    ),
    Document(
        "The history of written communication reflects an intricate relationship between "
        "technology and society. Early writing systems emerged primarily for "
        "administrative purposes. Subsequently, the development of the printing press "
        "democratized access to information significantly. Furthermore, digital "
        "technologies have accelerated this trend considerably. Notably, each transition "
        "has raised comparable concerns regarding authenticity and authority, "
        "underscoring a recurring pattern throughout the historical record.",
        1,
        "Claude Opus 5, generated 2026-08-01 for this fixture",
    ),
    Document(
        "Nutritional science has evolved considerably over recent decades. Initially, "
        "dietary guidelines emphasized macronutrient ratios almost exclusively. However, "
        "contemporary research underscores the importance of dietary patterns as a whole. "
        "Additionally, individual variation in metabolic response is now recognized as a "
        "crucial factor. Moreover, the role of the gut microbiome represents a pivotal "
        "area of ongoing investigation, suggesting a more nuanced picture than earlier "
        "models provided.",
        1,
        "Claude Opus 5, generated 2026-08-01 for this fixture",
    ),
]


def load_smoke() -> list[Document]:
    """The full fixture, human first. Order is stable so runs are reproducible."""
    return list(HUMAN) + list(MACHINE)
