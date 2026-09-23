# Prompt For Perplexity

You are helping revise an IEEEtran conference paper that must remain within a
strict 5-page limit, excluding references only if the venue allows references
outside the page count. The paper is about validation-locked calibration for
motor-imagery EEG/BCI decoding. Rewrite and tighten the draft without adding
unsupported claims.

Goals:

1. Keep the paper within 5 IEEE conference pages.
2. Preserve the core contribution: validation-locked model/calibration
   selection where held-out evaluation labels are final-report-only.
3. Update the Results and Discussion to include the new BNCI2014_004 external
   check.
4. Add or integrate statistical reporting: paired subject-level bootstrap 95%
   CIs, Wilcoxon signed-rank p-values, and win/tie/loss counts.
5. Make the claim more compelling but honest: BCI IV-2a supports seed
   ensembling; BCI IIIa is supportive but weaker due to stratified splitting;
   BNCI2014_001 is mixed/negative; BNCI2014_004 is strong above chance but
   mostly ties baseline accuracy, with teacher-student mixing slightly better
   on Brier/NLL and seed ensemble slightly better on ECE.

Current key numbers to integrate:

- BCI IV-2a seed ensemble vs baseline:
  - Brier delta -0.0195, 95% CI [-0.0319, -0.0054], Wilcoxon p = 0.027,
    W/T/L = 8/0/1.
  - Accuracy delta +0.018, 95% CI [0.007, 0.030], Wilcoxon p = 0.039,
    W/T/L = 6/1/2.
- BCI IIIa seed ensemble vs baseline:
  - Brier delta -0.0359, 95% CI [-0.0808, -0.0109], Wilcoxon p = 0.250,
    W/T/L = 3/0/0.
  - Accuracy delta +0.024, 95% CI [0.000, 0.071], Wilcoxon p = 1.000,
    W/T/L = 1/2/0.
- BNCI2014_001 seed ensemble vs baseline:
  - Brier delta +0.0005, 95% CI [-0.0165, 0.0188], Wilcoxon p = 0.734,
    W/T/L = 6/0/3.
  - Accuracy delta -0.008, 95% CI [-0.025, 0.009], Wilcoxon p = 0.570,
    W/T/L = 3/0/6.
- BNCI2014_004 held-out means:
  - Baseline accuracy 0.7866, Brier 0.2774, ECE 0.0669, NLL 0.4252.
  - Seed ensemble accuracy 0.7866, Brier 0.2786, ECE 0.0621, NLL 0.4276.
  - Teacher/student accuracy 0.7866, Brier 0.2739, ECE 0.0623, NLL 0.4147.
  - Augmentation accuracy 0.7822, Brier 0.2814, ECE 0.0709, NLL 0.4349.
  - Chance accuracy is 0.5. Trial-weighted accuracy for baseline, seed
    ensemble, and teacher/student is 2242/2840 = 0.7894; augmentation is
    2230/2840 = 0.7852.

Required edits:

- Replace any old statement that BNCI2014_004 is future work.
- Avoid saying the method broadly improves across datasets. Say transfer is
  mixed.
- Keep the aggregate table compact. Prefer one main aggregate metrics table
  plus one small paired-statistics table. Put extra ECE/NLL or per-method
  details into prose or appendix/supplement language if page space is tight.
- If space is tight, prioritize: Abstract, validation protocol, dataset audit,
  BCI IV-2a paired result, external mixed-transfer paragraph, limitations.
- Do not invent new experiments, citations, or numbers.
- If you suggest citations, mark them as suggestions and verify bibliographic
  details.

Desired framing:

The contribution is a leakage-aware, validation-locked calibration evaluation
for MI-BCI, showing that simple seed ensembling gives a reproducible primary
gain on BCI IV-2a but does not establish universal cross-dataset superiority.
BNCI2014_004 strengthens the external validation story by showing clean,
well-above-chance two-class performance under the same discipline, while also
showing that calibration/augmentation variants are not magic.

Please return:

1. A concise revised abstract.
2. A revised Results section that fits the 5-page constraint.
3. A revised Discussion/Limits paragraph.
4. A compact LaTeX table plan: which table(s) to keep, which numbers to omit
   or move to supplement.
5. A short checklist of exact phrases/claims to remove from the old draft.
