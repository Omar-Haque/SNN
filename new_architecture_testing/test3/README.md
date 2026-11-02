# test3: the reservoir as a plasticity reserve (class-incremental learning)

**The claim.** In this network a trained template is protected only because its neuron rarely fires: θ sits 25–35 mV
above base, so it wins only images that match it well, and STDP moves only the winner's weights. Stability comes from
the θ *level*. A flat neuron at θ_base wins the images nothing else matches and imprints them in one presentation.
Plasticity comes from the θ *gap*. During ordinary training that plasticity is harmful (the orphans belong to classes
that already have templates: hijacking, the plateau). When a genuinely **new class** arrives, the same machinery learns
it by **allocation** — a reserve neuron is recruited and captures the class — while the old templates never fire on the
new images and are shielded. Without a reserve a new class can only be learned by **overwriting** the best-overlapping
old templates: forgetting is the learning route itself. So: *the θ gap is the stability–plasticity dial, and the
reservoir we spent three rounds removing is its plasticity reserve.*

**The stream.** Task-free class-incremental MNIST: `--phases "0-4,5-9"`, two epochs per phase. No labels, no phase
signal — only the order of the images changes. Evaluation is unchanged (labels from the full 2,000-image assignment
set, all ten test classes scored at every snapshot), so new-class accuracy sits at chance until the network has
neurons for the new classes, and old-class accuracy is tracked throughout.

## Variations

| | knobs | isolates | prediction |
|---|---|---|---|
| **V1 no reserve** | `--init imprint --reserve 0` | learning without allocation (400 templates imprinted from phase-1 images; no flat neuron anywhere) | new classes learned slowly and by blurring old templates; old-class accuracy falls as new-class accuracy rises; old-class neurons flip label (F11 bottom: everything "repurposed") |
| **V2 fixed reserve** | `--reserve K` (100, 200) | allocation | new classes recruit reserve neurons within a few k examples of the boundary (F5 staircase restarts, F9 recruits jump); old-class accuracy preserved; new-class accuracy saturates with K |
| **V3 replenished** | `--reserve 100 --replenish` | sustained plasticity and the cost of recycling | handles repeated arrivals; recycling deletes the least-recently-fired templates, so forgetting is rate-limited and measurable (F9 recycled, `recycled` in the table) |
| **baseline** | `--init uniform --theta-plus 0.05` | the original network | a reserve by accident, with a weak conscience: learns the new classes through long hijacks and a second plateau |
| **θ decay** (`--theta-tau 30000`, crossed with V1/V2) | | the forgetting clock | a neuron whose class stops appearing loses its θ excess with time constant τ, becomes recruitable and gets overwritten: old-class accuracy should start to decay ≈ τ examples into phase 2; without decay it holds |

## How the reserve works (the "orphan rule")

Reserve neurons are flat (uniform rows) with θ **pinned** at θ_base while in the reserve. During training they are
**gated**: held at rest for the first `--reserve-delay` ms (default 10) of every presentation, and for the rest of the
presentation on every image on which a trained neuron has already fired. So a reserve neuron can only win an image
that no trained neuron claimed within the head start — an image that is novel to the trained population. The gate is
a per-image latch outside the model (the forward pass is verbatim from test2). A reserve neuron leaves the reserve the
moment its template becomes digit-like (top‑100 share ≥ 0.35, checked every batch) and is then an ordinary neuron.
Reserve neurons do not take part in evaluation (θ → +∞ for the snapshot, then restored), so accuracy always measures
the trained population.

Why a head start rather than θ alone: from the membrane equation a flat neuron reaches θ_base about 12 ms after
release; a trained neuron fires on its own class in ~10–20 ms and on a novel class in ~30–50 ms (both depend on its
θ level). The head start puts the reserve's latency between the two. **F10** measures those latencies directly on the
eval set (trained population, per class), so the first run tells you whether 10 ms is right: old-class latency should
be below the head start + 12 ms, new-class latency above it. Adjust `--reserve-delay` if not.

## Files

- `test3.py` — `test2.py` + the class-ordered stream, the reserve (gate, pin, recruitment, replenishment), the
  first-spike diagnostic and the extra saved arrays. Dynamics and estimators verbatim (AST-checked). Default 2 epochs
  per phase, θ⁺ 0.5, no decay, `--init imprint` (templates drawn from phase-1 images only).
- `test3_plots.py` — F1–F7 as before, plus **F8** per-class accuracy, **F9** reserve size / recruits / recycles,
  **F10** first-spike latency and silent fraction per class, **F11** label assignments over time and, for the new
  classes, allocated (ex-reserve) vs repurposed (ex-old-class) neurons. Phase boundaries as black lines.
- `summarize_test3.py` — `results/summary.csv` + S1 old/new accuracy curves, S2 the stability–plasticity plane (one
  point per run), S3 allocation vs repurposing bars.
- `run_test3.sh` — first pass (seed 0, ~3.5 h), summary, second pass (seed 1, ~2 h); five-arrival stream commented.

## Reading the results

- **Forgetting:** `acc_old_boundary` vs `acc_old_end` (S1 top; F8). V1 should drop, V2 should hold, V2+τ should
  hold for ~τ then drift.
- **Plasticity:** `acc_new_end` and `t_new` (S1 bottom). V2 should get there within a few k examples of the boundary;
  V1 slowly; the baseline through a plateau.
- **Mechanism:** F11 bottom and S3 — V2's new-class neurons should be ex-reserve, V1's ex-old-class.
  `repurposed_frac` is the neuron-level forgetting number the two variants should disagree on most.
- **Gate placement:** F10 at the boundary (`fs_old_b`, `fs_new_b` in the table).
- **The cost of V3:** `recycled` and which labels the recycled neurons carried (`recycle_events` in `run.npz`).

## How to run

```bash
cd new_architecture_testing/test3
python test3.py --smoke        # ~2 min; exercises the stream, gate, recruitment, replenishment, decay and all figures
bash run_test3.sh
python summarize_test3.py      # any time
python test3.py --replot results/<run>
```
