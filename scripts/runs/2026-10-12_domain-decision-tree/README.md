# domain-decision-tree (2026-10-12)

**Intent:** run a real 2-level Jev-style decision tree in PromptChain, in domains far from
billing/tech. Level 1 is new: "clinical or legal?". Level 2 reuses the bake-off's measured
questions (clinical specialty, 4 labels; legal clause type, 4 labels) and labelled items from
`harness-lab/bench/decisions/`. There are also 3 hand-written crossover texts meant to blur the root.
Thresholds 0.9 / 0.6 are placeholders.

**Triggered by:** Gyasi asked to see a chain built and run on domains other than billing/tech.

## Result (n=16 labelled, 2 per label; directional only)

| | tev1:4b | hunch mistral-small:22b |
|---|---|---|
| exact correct leaf | 14/16 | 10/16 |
| wrong leaf | 0 | 0 |
| top label correct (any band) | 15/16 | 16/16 |
| to review / escalate | 2 | 6 |
| crossovers flagged unclear | 3/3 at root | 1/3 at root |
| wall time (19 items) | 15.4 s | 56.0 s |

- `tev1` root confidence was 0.96–0.997 on clean items and 0.70–0.80 on all three crossovers, so
  each crossover went to review.
- `hunch` was right more often but less sure on legal clauses (0.22–0.88), so fewer items cleared
  0.9. Its root sat at 1.000 on nearly everything, including the malpractice clause (legal@0.98).
  The consent text read clinical@0.000, a flat spread. That fits the known plain-mode saturation.

Run: `python run.py --backend tev1|hunch` (needs the governed Mac lane; `hunch` on PATH).
