# test4: "use it or lose it" — idle neurons go back into the trainable reservoir

**Architecture 1.** A hijacker is a one-shot feature learner: flat template, θ at base, it wins what nothing else
captures and imprints it. The baseline trains every neuron that way and then runs out of such neurons. This variant
keeps the supply alive with no imprinting and no novelty gate: a trained neuron that has not spiked for
`--recycle-idle` examples (default 10k) is retired — template reset to a flat uniform row, θ to θ_base — and becomes
recruitable again. Recycling is budgeted: it happens only while the pool of flat neurons is smaller than `--pool`.
Everything is learned by STDP from a uniform init in the fast regime (θ⁺ 0.5, optional decay).

Predictions:
- **i.i.d. stream:** the conscience keeps nearly every neuron in use, so recycling should almost never trigger and
  nothing should change. (If it does trigger, those were dead units and recycling them should help slightly.)
- **class-incremental stream (0–4 then 5–9):** the old classes' neurons fall idle after the boundary; the longest-idle
  are recycled and recruited by the new classes. New classes learned by *allocation* of retired neurons; old classes
  forgotten only in proportion to the pool plus what the new classes recruit (budgeted forgetting) — in contrast to
  test3, where every variant overwrote 55–60 % of the old-class neurons in place.
- **what decides which neurons go:** use, not novelty. test3 showed no novelty signal exists in first-spike latency
  for this representation, so this is the mechanism that does not need one.

## What changed from test3
- No imprinting anywhere; uniform init only. No gate, no θ pinning by default (`--gate` restores test3's orphan rule
  for comparison; with the gate off the flat neurons take part in evaluation, as in the baseline).
- The "reserve" is the pool of flat neurons: all 400 at t = 0 (the natural reservoir), draining as they are recruited,
  topped up by recycling once below `--pool`.
- `--recycle-idle N`, `--neurons N` (network size), and **weight snapshots**: W at every phase boundary and at the end,
  so stability is measured in weight space (per-neuron cosine between the template at the boundary and at the end)
  rather than through the readout, whose re-assignment at every snapshot confounded old-class accuracy in test3.
- New figure **F12**: histogram of those cosines for the old-class neurons; `overwritten_frac` (cos < 0.9) and
  `cos_old_mean` in the summary are the forgetting numbers to look at first.

## Files
`test4.py` (dynamics and estimators verbatim; AST-checked), `test4_plots.py` (F1–F12), `summarize_test4.py`
(`summary.csv`, S1 old/new accuracy, S2 stability–plasticity plane, S3 allocated vs overwritten), `run_test4.sh`.

## How to run
```bash
cd new_architecture_testing/test4
python test4.py --smoke        # ~2 min; exercises both streams' code paths, recycling, decay, all figures
bash run_test4.sh              # ~5 h in three blocks; summarize_test4.py at any time
```
