# detbench

**A referee for AI-text detectors.**

This project does not claim to tell you whether a piece of writing was produced by a
machine. It measures the tools that claim that — including its own — at the operating
points that matter, and publishes where they fail.

That distinction is the entire point, so it is worth being blunt about why.

## Why a referee and not another detector

Every detector on the market leads with a single number, usually AUROC, usually above
0.95. That number is close to useless for the decision people actually make with these
tools, which is *"do I accuse this person?"*

AUROC averages performance across every possible threshold, including thresholds nobody
would ever deploy. A detector can post an excellent AUROC while being catastrophically
wrong at the only setting that matters. This is not a hypothetical: on a leave-one-domain-out
evaluation, one recent study found **60.4% of human-written documents were assigned
near-certain machine labels** (p ≥ 0.95) by a detector whose headline AUROC looked fine.
Six out of ten innocent people, flagged with confidence.

So `detbench` reports three numbers first, and AUROC last:

| Number | The question it answers |
|---|---|
| **TPR @ 1% FPR** | How much machine text do you catch if you accept wrongly accusing 1 in 100 people? |
| **TPR @ 0.1% FPR** | And at 1 in 1,000? |
| **Human called machine (p ≥ 0.95)** | How much human writing does this tool feel *certain* about, wrongly? |
| **Refusal rate** | How often did it decline to answer — and is its accuracy just cherry-picked easy documents? |
| AUROC | Reported for comparability. Do not decide on it. |

The third row is the one almost nobody publishes. It is also the one that corresponds to
a real person being harmed.

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"

detbench list                    # what's registered
detbench demo                    # the whole argument in five seconds
detbench score "..." --detector stylometric
```

`detbench demo` runs on a twelve-document fixture and prints a table full of `n/a`.
**That is the demo.** Six human documents cannot resolve a 0.1% false-positive rate — the
smallest rate you could even observe is 17% — so the harness refuses to print a number
rather than inventing one. Most published detector claims are built on samples not much
larger.

The core package has **no dependencies**. Metrics, attacks, and the stylometric baseline
run on a bare Python 3.11. Model-based detectors are an optional extra
(`pip install detbench[torch]`), and if torch is missing they *refuse* rather than fail.

## What's in the box

**Detectors**

- `stylometric` — twelve interpretable surface features, no model, runs anywhere. Weak on
  clean text and included precisely for that reason: under paraphrase attack the published
  ordering inverts, and the weak-but-stable method beats the strong-but-fragile one.
- `binoculars` — perplexity ÷ cross-perplexity (Hans et al., ICML 2024). State of the art
  on clean text; the largest measured F1 drop of any method under paraphrase.
- `fast-detectgpt` — conditional probability curvature, analytic single-pass form.

**Attacks** (all deterministic under a seed, so every published row is reproducible)

- `homoglyph` — Cyrillic and Greek lookalikes. Invisible to a reader, devastating to a
  tokenizer, and undone by a *confusables fold* that most deployed detectors do not make.
  Note that NFKC normalisation alone does **not** fix this — Cyrillic `а` and Latin `a`
  are distinct characters, not compatibility variants, and NFKC leaves them untouched.
  This README asserted otherwise until the test suite caught it.
- `zero_width` — invisible characters between words. The cheapest possible attack, and
  therefore the floor.
- `synonym` — a small hand-checked substitution table. Deliberately weaker than a real
  paraphraser: if this floor already moves a detector, the detector is fragile.

**Preprocessing sensitivity** (`detbench sensitivity [files...]`)

Attacks measure what an adversary does to you. This measures what *you* do to yourself —
the pipeline choices nobody writes down. Strip the code blocks or don't; drop the front
matter or don't. On six real technical documents, the usual composite (`prose_only`) moved
scores by a mean of **0.270**, against a human/machine signal gap of 0.151 for the same
detector. Suggestive, not conclusive — those come from different document sets, so the tool
prints `n/a` for the ratio rather than dividing across them.

The point stands regardless of the exact figure: if an unremarked plumbing decision moves
the answer as far as the signal does, then "we ran detector X" is not a reproducible claim.
`collapse_whitespace`, for contrast, is provably free — zero shift on every document.

## Two rules the code enforces, not just documents

**Refusal is a result.** A detector may return "I decline" instead of a score, and short
text triggers that automatically. The type system makes a refused verdict incapable of
carrying a number, so a refusal cannot quietly be used as a zero.

**A score is not a probability.** `p_machine` stays `None` until a detector has been
calibrated on labelled data, and any probability that does come back carries the name of
the distribution it was calibrated against. Converting a raw score into a confident-looking
percentage is the single most common way this category misleads people, and here it is a
type error rather than a style guideline.

## What this project does not claim

- **It cannot reliably tell you if a specific document was AI-written.** Nothing can, at
  the confidence levels people want. As model output distributions converge on human
  writing, the theoretical ceiling on any detector approaches chance.
- **Detection failures are not evenly distributed.** False positives fall hardest on
  non-native English writers and on formulaic human prose. A tool that is 99% accurate
  overall can still be systematically wrong about one group of people.
- **The included detector implementations are not yet validated** against their reference
  implementations. Until `scripts/validate.py` reproduces published numbers on a shared
  slice, treat any output as indicative. This is a release blocker for publishing a
  leaderboard, and it is tracked as such.
- **The shipped fixture is a smoke test, not an evaluation set.** Real runs use RAID,
  MAGE, and PADBen — see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Status

Early. The contracts, metrics, attacks, harness, and the no-model baseline are implemented
and tested. The model-bearing detectors are implemented but unvalidated. There is no
leaderboard yet, because publishing one before validation would be exactly the behaviour
this project exists to criticise.

## License

MIT.
