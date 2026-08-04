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

### Full run, all three detectors (2026-08-03)

3,000 documents (1,500 human / 1,500 machine), 12 generators, 8 English prose domains,
seed 0, GPU, float32. Raw output in `results/raid_full.txt`.

| detector | slice | TPR@1%FPR | TPR@0.1%FPR | AUROC |
|---|---|---:|---:|---:|
| stylometric | clean | 38.7% | 11.7% | 0.756 |
| binoculars | clean | 49.7% | 38.4% | 0.781 |
| fast-detectgpt | clean | **51.3%** | **38.7%** | **0.787** |
| stylometric | homoglyph | 36.5% | 9.6% | 0.738 |
| binoculars | homoglyph | 20.4% | 8.5% | 0.734 |
| fast-detectgpt | homoglyph | 23.4% | 13.7% | 0.616 |
| stylometric | zero_width | **36.6%** | **10.6%** | **0.744** |
| binoculars | zero_width | 14.7% | 7.7% | 0.618 |
| fast-detectgpt | zero_width | 18.1% | 10.6% | 0.548 |
| stylometric | synonym | 38.7% | 11.7% | 0.756 |
| binoculars | synonym | 47.9% | 36.2% | 0.777 |
| fast-detectgpt | synonym | 50.0% | 37.8% | 0.783 |

**The ordering inverts, and that is the result.** On clean text the ranking is
fast-detectgpt > binoculars > stylometric. Under `zero_width` it is
stylometric > fast-detectgpt > binoculars. **The weakest detector on clean text is the
strongest under the cheapest available attack.**

`zero_width` inserts invisible characters between words. It costs an evader seconds, leaves
the rendered document byte-identical to a reader, and takes Fast-DetectGPT from AUROC 0.787
to **0.548** — within a whisker of a coin flip — while the stylometric baseline moves 0.756
to 0.744. This independently reproduces the accuracy/robustness inversion reported for
paraphrase attacks in arXiv:2605.14240, except that the attack here is *free* rather than
requiring a paraphrase model.

Two further observations:

- **Even undisturbed, the best detector catches under 40% of machine text** at a
  false-positive rate you could defend, while its AUROC reads 0.787. The gap between those
  two numbers is the reason this project reports the operating point first.
- **The synonym attack does almost nothing to anything** (≤1.4 points). Our synonym table is
  deliberately tiny, so this is a floor, not evidence that lexical substitution is harmless.

**Load-bearing caveats.**

1. **These are GPT-2-scale models, not the reference configurations.** Binoculars is
   `gpt2`/`gpt2-medium`, not the Falcon-7B pair; Fast-DetectGPT is single-model `gpt2-medium`,
   not `gpt-j-6B`→`gpt-neo-2.7B`. Treat every number as a floor for the method, not as the
   published performance of it.
2. **No input normalisation is applied, and that is the whole story for the Unicode rows.**
   `attacks.lexical.normalize_unicode` neutralises both `homoglyph` and `zero_width`
   completely. These rows measure *undefended* detectors. A defended-variant slice is the
   obvious next piece of work, and until it exists this table should be read as "what happens
   if you do not normalise", not "these methods are broken".
3. **The run hit GPU memory pressure.** A CUDA caching-allocator OOM warning appears in the
   Binoculars section of `results/raid_full.txt`. The allocator recovered; the results were
   checked afterwards against CPU on the same documents (worst difference **8.7e-06**) and
   refusal rates are identical across all three detectors (0.6%), so nothing was dropped or
   corrupted. Recorded because the warning is in the raw output and a reader will see it.
4. **GPU float32 is not bit-identical to the validated CPU float32.** Measured deltas are
   5.8e-06 (Binoculars) and 1.6e-04 (Fast-DetectGPT) — the latter slightly exceeding the 1e-4
   tolerance the validation scripts use. cuBLAS non-determinism, not a defect, but it means
   these numbers are not the exact ones equivalence was established at.

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

1. **Binoculars is validated at the algorithm level; Fast-DetectGPT is not validated at
   all.** See § 6 below — the distinction between "computes the right quantity" and
   "reproduces the paper's numbers" is doing real work there, and only the first is done.
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

## 6. Validation status

Two different claims get conflated as "validated", so this section separates them.

**(a) Algorithmic equivalence** — does our code compute the same quantity as the reference?
**(b) Reproduction** — do we get the paper's reported numbers?

| Detector | (a) equivalence | (b) reproduction |
|---|---|---|
| `binoculars` | ✅ 2026-08-02, agreement to **2.9e-08** | ❌ blocked on hardware |
| `fast-detectgpt` | ✅ 2026-08-03, exact + Monte-Carlo cross-check | ❌ blocked on hardware |
| `stylometric` | n/a — ours, no reference exists | n/a |

### Binoculars: what the check does

`scripts/validate_binoculars.py` transcribes `binoculars/metrics.py` and
`binoculars/detector.py` from ahans30/Binoculars verbatim and runs it as an **independent
code path** against ours on the same inputs. A shared bug would have to be transcribed
twice to survive. Worst absolute difference across four documents spanning 19th-century
prose, model-shaped prose, technical writing and code: **2.9e-08**.

Run it with `python scripts/validate_binoculars.py`.

### What validation caught, which is the reason it is a release blocker

Our implementation was **wrong in two substantive ways**, and both produced entirely
plausible numbers:

1. **The numerator took perplexity from the observer model.** The reference takes it from
   the **performer**; the observer appears only inside the denominator, as the distribution
   the performer is scored against. Easy to invert, because the paper's prose describes the
   observer as the model that "computes the perplexity of the text."
2. **The cross-entropy term was computed on shifted logits.** The reference does not shift
   it — the numerator runs over T−1 positions and the denominator over T. They deliberately
   do not align, which reads like an oversight until you match it and the numbers agree.

Combined error on real text: **4.6% to 12.4%**. For scale, the gap between the reference's
own published thresholds — 0.8536 (low-FPR) and 0.9015 (accuracy) — is 5.6%. The error was
larger than the distance between the two operating modes, so it could flip a verdict.

A third gap was closed at the same time: the reference asserts both models share a
vocabulary and raises otherwise. Ours loaded only the observer's tokenizer, so a mismatched
pair would have produced a confident, meaningless number. `tests/test_binoculars.py` pins
all of this with synthetic logits so it runs offline.

### Fast-DetectGPT: why exact agreement was the *weaker* result

`scripts/validate_fast_detectgpt.py` reports agreement of **0.00e+00** against the reference
criterion, on both the shared-model and two-model configurations. That number looks better
than Binoculars' 2.9e-08 and means less.

Our criterion is a near-verbatim transcription of the reference's, so identical arithmetic
in identical order produces bit-identical floats. **A transcription check cannot detect an
error that was transcribed.** It validates the surrounding pipeline — tokenization,
shift alignment, the two-model branch, dtype handling — and nothing about the mathematics.

So the real check compares the analytic criterion against the reference's own **Monte-Carlo**
criterion, the one the closed form replaces. If the derivation or our port of it were wrong,
the two would diverge by far more than sampling error. Convergence on a single document,
seed 0:

| samples | Monte-Carlo | abs diff from analytic (0.383000) |
|---:|---:|---:|
| 250 | 0.324301 | 0.0587 |
| 1,000 | 0.370584 | 0.0124 |
| 4,000 | 0.369727 | 0.0133 |
| 16,000 | 0.378378 | 0.0046 |
| 64,000 | 0.382348 | **0.0007** |

Clean 1/√N convergence to the analytic value. That is an independent confirmation of the
closed form, and it is the only part of this detector's validation that could have failed.

### What validation changed here

Less dramatic than Binoculars, but not nothing:

1. **Two-model support was missing.** The reference separates the *sampling* model (which
   supplies the distribution the expectation is taken over) from the *scoring* model. Ours
   supported only the shared case and described it as "the white-box setting used here" —
   accurate, but it meant the detector could not run in the configuration the published
   numbers come from (`gpt-j-6B` sampling, `gpt-neo-2.7B` scoring).
2. **Per-position variance was clamped** to 1e-12. The reference does not clamp. Whenever
   floating-point error produced a small negative term, the clamp silently changed the sum
   and made this a different statistic. Removed; degenerate *totals* are now surfaced as a
   refusal at the detector level instead, which does not alter any non-degenerate value.
3. **The reference's fitted calibration is now recorded and used.** Their `local_infer.py`
   publishes normal-distribution parameters per model pair, which convert a raw curvature
   into a probability. `p_machine` is reported when the configured pair has published
   parameters and stays `None` otherwise — the same calibration discipline as everywhere
   else in this package, satisfied for the first time by someone else's fitted constants.

### What is still NOT established

The published results — >90% TPR at 0.01% FPR, state-of-the-art zero-shot AUROC — were
measured with **Falcon-7B / Falcon-7B-Instruct at bfloat16**, roughly 28 GB of weights
across the pair. The machine this was developed on has a 4 GB GPU and 15 GB of RAM, so that
configuration cannot run here. Equivalence was therefore confirmed on `gpt2` / `gpt2-medium`,
which shares the reference's tokenizer-consistency requirement but is **not** the reference
configuration.

So: **the implementation is checked, the paper's results are not re-derived.** Different
claims. A leaderboard row may cite our own measured numbers once the remaining detectors are
equivalence-checked; it may **not** cite the paper's headline figures as though we had
reproduced them.

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
