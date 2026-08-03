# Methodology

How this benchmark measures, why it measures that way, and what it currently cannot claim.

## 1. The operating point is the whole argument

A detector is not deployed at "all thresholds simultaneously", which is what AUROC
averages over. It is deployed at one threshold, chosen by whoever runs it, and the only
question that matters at that threshold is: *how many innocent people does this accuse to
catch how much machine text?*

So the primary metric is **true-positive rate at a fixed false-positive rate**, reported
at 1% and 0.1%. Two consequences follow, and both are enforced in code rather than
recommended in prose.

**Sample size gates the claim.** You cannot observe a 0.1% false-positive rate with fewer
than 1,000 human documents, because the smallest non-zero rate a sample of *n* can express
is 1/*n*. `metrics.threshold_at_fpr` returns `None` when the sample cannot resolve the
requested rate, and the table prints `n/a`. A detector page quoting "0.01% false positives"
from a few hundred documents is quoting a number its data cannot contain.

**Ties must not be allowed to inflate the number.** Scoring uses `score >= threshold`. If
the threshold lands inside a block of human documents tied at the same score, every one of
them is admitted, and the realised false-positive rate can be orders of magnitude above the
budget while the label on the chart still reads "1%". `threshold_at_fpr` walks distinct
values upward and picks the lowest threshold whose *realised* count stays in budget. This
was a live bug in the first implementation here, caught by
`test_tpr_at_fpr_penalises_overlap`, and it inflated TPR from 0.5 to 1.0 — a 100% relative
overstatement from a two-line mistake.

## 2. The number nobody publishes

`overconfidence_rate` is the fraction of **human** documents a calibrated detector assigned
p ≥ 0.95 of being machine-generated. It corresponds directly to a person being confidently
and wrongly accused.

The motivating result: on a leave-one-domain-out evaluation, 60.4% of human-written samples
received near-certain machine predictions (arXiv:2607.03680). A detector can hold a strong
AUROC while doing this, because AUROC is rank-based and never looks at whether the
probabilities mean anything.

This metric is `None` for uncalibrated detectors, which is most of them, and that `n/a` is
itself informative: a detector that reports a percentage without having been calibrated is
reporting a number with no defined meaning.

## 3. Attacks, and why the weak ones are included

Clean-text accuracy is the number a detector advertises. Post-attack accuracy is the number
that decides whether it works. Published results show these diverge sharply, and — this is
the important part — **they diverge most for the best detectors**. Of seven methods tested
against paraphrase, Binoculars had both the strongest clean-text performance and the
largest degradation, losing 0.196 F1; a plain text-feature model lost 0.053
(arXiv:2605.14240).

The attacks shipped here are deliberately *weaker* than a determined evader's:

| Attack | Cost to the evader | Why it is included |
|---|---|---|
| `zero_width` | seconds | The floor. Failing here means failing against no effort at all. |
| `homoglyph` | seconds | Tests whether a deployment normalises input. Most do not. |
| `synonym` | seconds | Isolates lexical from structural evasion. |

If a detector already moves under these, a real paraphrase attack is not a question worth
asking. Model-bearing paraphrase (DIPPER-class, or an LLM rewrite) is the obvious next
addition and is not yet implemented.

Attacks are applied to **machine text only**. An evader has no reason to launder text that
was already their own, and attacking human text would let post-attack scores improve for
reasons unrelated to evasion.

Every attack is deterministic under a seed, so a published row is reproducible exactly.
`AttackResult.edit_rate` is recorded so that an attack which "succeeds" by destroying the
document can be spotted: evasion that rises together with a high edit rate is a finding
about text destruction, not about the detector.

## 3b. Preprocessing sensitivity — the pipeline nobody writes down

Attacks measure what an adversary can do to you. This slice measures what **you** do to
yourself, and it is a separate module (`preprocessing.py`, `sensitivity.py`) rather than an
attack, because the two answer different questions. An attack is adversarial and applied to
machine text only. A preprocessor is a defender's choice applied uniformly to everything:
strip the code blocks or don't, drop the front matter or don't, collapse the whitespace or
don't. Nobody reports these choices, because they feel like plumbing.

The headline is:

    sensitivity_ratio = mean |score(variant) - score(raw)|  /  |mean(machine) - mean(human)|

Above **1.0**, an undocumented pipeline decision moves the score further than the signal the
detector is supposed to be measuring. At that point "we ran detector X" stops being a
reproducible statement: two people can run the same detector on the same document, make
different unremarked choices, and disagree by more than the human/machine difference.

**First measured result (2026-08-02), on six real technical documents:**

| Preprocessor | mean \|shift\| | max \|shift\| |
|---|---:|---:|
| `prose_only` (the usual composite) | **0.270** | **0.455** |
| `strip_markdown` | 0.142 | 0.298 |
| `strip_code` | 0.087 | 0.318 |
| `strip_urls` | 0.022 | 0.086 |
| `strip_front_matter` | 0.011 | 0.041 |
| `collapse_whitespace` | 0.000 | 0.000 |

For scale, the same detector's human/machine signal gap on the labelled fixture is
**0.151**. **This does not license the ratio it invites.** The two numbers come from
different document sets — twelve short prose samples versus six long markdown files — and
dividing across sets is not a valid ratio. The tool refuses to print one for exactly this
reason: `sensitivity_ratio` is `None` without labels, and the unlabelled table shows `n/a`.
The comparison is suggestive and nothing more. Obtaining a real ratio requires a labelled
corpus of markdown-shaped documents, which is another reason to wire RAID.

Two secondary findings the slice produces for free:

- **`collapse_whitespace` is provably free.** Zero shift, every document. Not every pipeline
  choice is dangerous, and a benchmark that implied otherwise would be crying wolf.
- **A preprocessor can push a document below the refusal gate.** Stripping code from a
  code-heavy document can leave too little prose to judge, so a detector that scored it
  before will now decline. This surfaces as `newly refused` in the table — the same document
  and the same tool giving a different answer about whether an answer is possible at all.

## 4. Corpora

The fixture shipped in `detbench.data.fixtures` is a **smoke test with twelve documents**
and supports no conclusion about anything. Its labels are exact — the human documents are
public-domain works from 1813–1859, the machine documents were generated by Claude Opus 5
on 2026-08-01 for this purpose — but twelve documents is a demo, not evidence.

Real evaluation uses:

- **RAID** (arXiv:2405.07940) — 6M+ generations, 11 generators, 8 domains, 11 adversarial
  attacks, 4 decoding strategies. The reference robustness benchmark.
- **MAGE** — used specifically for its leave-one-domain-out split, which is where
  distribution-shift failure becomes visible.
- **PADBen** (arXiv:2511.00416) — iterated paraphrase, with a five-type taxonomy covering
  the path from original text to deeply laundered text.

**RAID is wired up** (`detbench.data.raid`, 2026-08-02). MAGE and PADBen are not.

### Using RAID

```bash
detbench raid-fetch --split train      # 802 MB, cached under ~/.cache/detbench
detbench raid --limit 2500             # 2,500 documents per class
```

Three defaults are methodology choices rather than conveniences, and each is something a
caller could get silently wrong:

- **English prose only.** RAID includes a `code` domain — the human rows are literally
  Python source — and two non-English domains (`czech`, `german`). Scoring source code with
  a detector whose features are sentence-length variance and discourse markers produces a
  number with no meaning, and given that false positives already concentrate on non-native
  English writers, silently mixing languages is worse than useless. Both are opt-in.
- **Non-adversarial by default.** RAID ships pre-attacked rows and `detbench` applies its
  own attacks; loading both would measure a compound transform nobody deploys. When you do
  opt in, the overlap is useful: RAID's `homoglyph`, `zero_width_space` and `synonym` are
  reference implementations of three attacks here, so the same source documents can check
  ours against theirs.
- **Balanced classes.** RAID's natural ratio is roughly 1 human document to 11 machine,
  because each human text seeds a generation from every model. Reporting a false-positive
  rate off that ratio would rest on a small human sample while looking like a large study.

**A trap worth naming: the `extra` split is not "more of the same."** Verified against the
real files — it contains *only* `code` (32,200 rows), `german` (68,950) and `czech`
(68,775). The eight English prose domains are all in `train`. Loading `extra` with the
default prose filter correctly matches nothing, so `load` raises with the domains actually
present rather than returning an empty list.

### First result on real data (2026-08-02)

`stylometric`, 5,000 documents (2,500 human / 2,500 machine), 12 generators, 8 English
prose domains, seed 0:

| slice | refused | TPR@1%FPR | TPR@0.1%FPR | AUROC |
|---|---:|---:|---:|---:|
| clean | 0.7% | **41.1%** | **29.3%** | 0.766 |
| homoglyph | 0.7% | 38.0% | 26.3% | 0.746 |
| zero_width | 0.7% | 39.4% | 26.4% | 0.755 |
| synonym | 0.7% | 41.1% | 29.3% | 0.766 |

**This single row is the argument the whole metric design was built to make.** An AUROC of
0.766 reads as a usable tool. Catching 29.3% of machine text at a false-positive rate you
could defend to the person being accused does not. Same detector, same documents, two
completely different impressions — and only one of them is the number anyone publishes.

Secondary observations: the synonym attack has *zero* effect on a feature-based detector
(41.1% → 41.1%), consistent with the published finding that text-feature models are the
most paraphrase-resilient; the two Unicode attacks cost 2–3 points, which is the tokenizer
being perturbed rather than the style; and this is a **weak** detector, so these numbers are
a floor for the benchmark, not a claim about the field.

### Correction: the below-chance result was a fixture artifact

An earlier revision of this document recorded that the unfitted stylometric baseline scored
**below chance** (AUROC 0.444, signal gap −0.151) and inferred that its hand-assigned
feature signs were "wrong-signed on net." **On 5,000 real documents it scores 0.766.** The
signs are fine; twelve documents were not enough to tell, and the conclusion drawn from them
was wrong.

That correction is left in rather than quietly edited out, because it is the clearest
possible demonstration of the point the fixture warnings were already making — including
against the person who wrote them.

## 5. What this benchmark cannot currently claim

Stated plainly, because a benchmark that hides its own limits has no standing to measure
anyone else's.

1. **The model-bearing detectors are unvalidated.** `binoculars.py` and
   `fast_detectgpt.py` implement the published algorithms, but neither has been checked
   against its reference implementation. Until `scripts/validate.py` reproduces published
   AUROC on a shared slice, no number from either may appear in anything public. **This is
   a release blocker, not a nice-to-have.**
2. **No real corpus is loaded.** Everything above describes how the harness measures; it
   has not yet measured anything at scale.
3. **The unfitted stylometric baseline currently scores below chance** on the smoke
   fixture (AUROC 0.444, and 0.361 under homoglyph attack). The hand-assigned feature
   signs in `HEURISTIC_SIGNS` are, on this evidence, not merely weak but wrong-signed on
   net. This is recorded rather than tuned away, for two reasons: tuning signs against a
   twelve-document fixture would be fitting to noise, and the result is a fair sample of
   what happens when anyone assigns feature weights by intuition — which is what a
   surprising number of shipped detectors do. The fitted path
   (`StylometricDetector.fit`) separates the fixture perfectly, which proves only that
   the fitting loop works and is *not* evidence of generalisation.
4. **No non-native-speaker slice yet.** Published work finds false positives concentrate
   heavily on non-native English writers. A benchmark that leads on false-positive harm
   and does not measure this specific harm is incomplete, and this is the highest-priority
   gap in the list.

## References

- Hans et al., *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated
  Text*, ICML 2024. arXiv:2401.12070
- Bao et al., *Fast-DetectGPT: Efficient Zero-Shot Detection via Conditional Probability
  Curvature*
- Dugan et al., *RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text
  Detectors*. arXiv:2405.07940
- *Rethinking AI-Generated Text Detection: A Strong Baseline and the Distribution-Shift
  Problem That Remains*. arXiv:2607.03680
- *Paraphrasing Attack Resilience of Various AI-Generated Text Detection Methods*.
  arXiv:2605.14240
- *PADBen: A Comprehensive Benchmark for Evaluating AI Text Detectors Against Paraphrase
  Attacks*. arXiv:2511.00416
- Sadasivan et al., *Can AI-Generated Text be Reliably Detected?*
