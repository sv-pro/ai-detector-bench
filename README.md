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

For real numbers, use RAID (Dugan et al., ACL 2024 — 6.2M generations, 11 generators,
11 domains, MIT licensed):

```bash
detbench raid-fetch --split train   # 802 MB, cached under ~/.cache/detbench
detbench raid --limit 2500          # 2,500 documents per class
```

**The result, on 3,000 RAID documents across 12 generators and 8 domains:**

| detector | clean | homoglyph | zero_width | synonym |
|---|---:|---:|---:|---:|
| | *AUROC / TPR@0.1%FPR* | | | |
| stylometric | 0.756 / 11.7% | 0.738 / 9.6% | **0.744 / 10.6%** | 0.756 / 11.7% |
| binoculars | 0.781 / 38.4% | 0.734 / 8.5% | 0.618 / 7.7% | 0.777 / 36.2% |
| fast-detectgpt | **0.787 / 38.7%** | 0.616 / 13.7% | 0.548 / 10.6% | 0.783 / 37.8% |

**The ordering inverts, and that is the finding.** On clean text: fast-detectgpt > binoculars
> stylometric. Under `zero_width`: stylometric > fast-detectgpt > binoculars. **The weakest
detector on clean text is the strongest under the cheapest available attack.**

`zero_width` inserts invisible characters between words. It costs seconds, leaves the
document visually identical, and drops Fast-DetectGPT from 0.787 AUROC to **0.548** — a coin
flip — while the twelve-feature baseline moves 0.756 → 0.744.

And note the clean row: the best detector here catches **under 40%** of machine text at a
defensible false-positive rate while posting a 0.787 AUROC. That gap is why the operating
point is reported first.

Two caveats that change how you should read this. These are GPT-2-scale models, not the
reference Falcon-7B configurations — floors for each method, not published performance. And
**no input normalisation is applied**: `normalize_unicode` neutralises both Unicode attacks
entirely, so those rows measure *undefended* detectors. Full detail and the remaining
caveats in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md); raw output in `results/`.

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
- **Binoculars is validated at the algorithm level, not reproduced.** `scripts/validate_binoculars.py`
  runs a verbatim transcription of the reference as an independent code path and agrees to
  **2.9e-08**. That confirms we compute the right quantity. It does **not** reproduce the
  paper's >90% TPR at 0.01% FPR, which needs the Falcon-7B pair at bfloat16 (~28 GB) and a
  GPU this project does not have. Two different claims; only the first is done.
  **Fast-DetectGPT is now validated too** — and its *exact* (0.00e+00) agreement is the
  weaker result, because our criterion is a transcription and a transcription check cannot
  detect an error that was transcribed. The check that counts compares the analytic form
  against the reference's Monte-Carlo form, which converges to it at 1/√N (0.0587 → 0.0007
  from 250 to 64,000 samples). Neither detector's published AUROC is reproduced.
- **Validation found our Binoculars was wrong twice** — perplexity taken from the observer
  instead of the performer, and a cross-entropy term shifted when the reference does not
  shift it. Both produced plausible numbers that were 4.6–12.4% off, more than the 5.6% gap
  between the reference's own two published thresholds. This is why unvalidated detectors
  are barred from the leaderboard rather than merely flagged.
- **The shipped fixture is a smoke test, not an evaluation set.** RAID is wired up; MAGE
  and PADBen are not — see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).
- **This project has already been wrong once in public, and left the evidence in.** An
  earlier revision reported the stylometric baseline scoring *below chance* and concluded
  its feature signs were inverted. On real data it scores 0.766. Twelve documents were not
  enough to tell, and the conclusion drawn from them was wrong — which is the point the
  fixture warnings were making, demonstrated against their own author.

## Status

Early. The contracts, metrics, attacks, harness, and the no-model baseline are implemented
and tested. The model-bearing detectors are implemented but unvalidated. There is no
leaderboard yet, because publishing one before validation would be exactly the behaviour
this project exists to criticise.

## License

MIT.
