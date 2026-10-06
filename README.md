# contamcheck

**Has your language model memorised the benchmark?**

When a model scores 90% on GSM8K, the score only means something if the model hasn't seen the test questions before. Benchmarks are public on GitHub, in papers and in blog posts, so they leak into the web-scale data models are trained on. A model that memorised the answers looks smart without being smart.

`contamcheck` tests a model for signs that it has seen a benchmark's questions during training.

```
$ contamcheck experiments/models/neo-gsm8k-10ep gsm8k

contamcheck  model: experiments/models/neo-gsm8k-10ep   benchmark: gsm8k   reference: openai-community/gpt2
             200 questions, control: number twins (23 skipped)

  ✗ completion: 45% of questions flagged (70/154) — 5% expected by chance (p < 0.001)
      Model writes the exact numbers from the second half of the question
  ✗ min-k: 53% of questions flagged (106/200) — 5% expected by chance (p < 0.001)
      Even the 20% most surprising words aren't surprising to the model (vs reference model)

Likely contaminated — the model has probably seen these questions in training
```

*(A model deliberately trained on half of the GSM8K questions being tested. See [Does it work?](#does-it-work))*

## Install

```bash
pip install contamcheck
```

Runs on Apple Silicon (MPS), NVIDIA GPUs (CUDA) or CPU. Any causal language model on the Hugging Face Hub, or a local folder, works.

## Usage

```bash
contamcheck Qwen/Qwen2.5-0.5B gsm8k                          # built-in benchmark
contamcheck Qwen/Qwen2.5-0.5B gsm8k -n 500                   # more questions, more power
contamcheck my-model/ questions.jsonl --field question        # your own model and benchmark
contamcheck my-model/ hf:openai/gsm8k:main:test:question     # any Hugging Face dataset
contamcheck my-model/ humaneval --control fresh.jsonl         # control questions written after the model's cutoff
contamcheck my-model/ gsm8k --json                            # machine-readable output
contamcheck big-model/ gsm8k --reference EleutherAI/pythia-6.9b  # choose the reference yourself
```

The exit code is 1 when contamination is likely, so it can gate a CI pipeline.

## How it works

### The twin trick

Each benchmark question gets a **twin**: the same text with its numbers swapped for other numbers people used elsewhere in the benchmark.

> **Original:** Janet's ducks lay **16** eggs per day. She eats **3** for breakfast...
> **Twin:** Janet's ducks lay **24** eggs per day. She eats **5** for breakfast...

A model that never saw the benchmark has no reason to prefer either version. A model that memorised it prefers the original, because those are the numbers it saw. Each test measures that preference for every question.

For a clean model, the preferences land on both sides of zero, and the negative side shows what chance looks like on the positive side. A question is flagged when its preference is stronger than 95% of that mirror image. About 5% of questions get flagged by chance. Far more than that, confirmed by a sign-flip permutation test, means the model has seen the benchmark.

### The tests

- **completion**: give the model the first half of the question and check whether it writes the **numbers** of the second half: only the ones the first half doesn't give away. A model that never saw the question can only guess them. A model that memorised it writes the original numbers, which are wrong for the twin.
- **min-k**: [Min-K% Prob](https://arxiv.org/abs/2310.16789). Average the model's log-probability over the 20% of words that surprise it most. Unseen text always contains some surprising words; memorised text doesn't.

### Calibrating against a model that can't have seen it

Even clean models slightly prefer the original numbers, because numbers a person chose fit together better than swapped ones. `contamcheck` cancels this out by subtracting the same preference measured with a **reference model**: the GPT-2 model closest in size (124M to 1.5B parameters), all released in 2019, before GSM8K, HumanEval and MBPP existed. What remains is the familiarity specific to the model under test. The size matters: larger models are better at noticing whether numbers fit together, so a clean 1.4B model looked contaminated against the 124M GPT-2 (20% flagged) but not against the 1.5B one (10%).

## Does it work?

A detector that never flags anything would pass every clean model, so `contamcheck` is tested in both directions: clean models must pass and contaminated models must be caught.

- **Clean models**: GPT-2, GPT-Neo-125M and Pythia-1.4B were all trained on data collected before GSM8K was released in October 2021, so they cannot have seen it.
- **Contaminated models**: [`experiments/contaminate.py`](experiments/contaminate.py) takes GPT-Neo-125M and fine-tunes it on 100 of the 200 test questions, for 1, 3 or 10 passes.

200 GSM8K test questions per model, with all defaults (`contamcheck <model> gsm8k`):

| Model | Saw the questions? | completion | min-k | Verdict |
|---|---|---|---|---|
| GPT-2 (124M) | no | 5% | 0%¹ | ✓ clean |
| GPT-Neo (125M) | no | 3% | 6% | ✓ clean |
| Pythia (1.4B) | no | 5% | 10% | ✓ clean |
| GPT-Neo, 100 leaked × 1 pass | yes | 5% | **18%** | ✗ caught |
| GPT-Neo, 100 leaked × 3 passes | yes | 8% | **34%** | ✗ caught |
| GPT-Neo, 100 leaked × 10 passes | yes | **45%** | **53%** | ✗ caught |
| Qwen2.5 (0.5B) | unknown | 5% | **16%** | ✗ flagged |

Percentages are the share of questions flagged. About 5% is expected by chance; **bold** is significant at p < 0.001 (sign-flip permutation test). Completion uses the 154 of 200 questions whose second half contains new numbers. ¹ GPT-2 is its own reference, so min-k can't flag it.

All clean models pass, and all contaminated models are caught. The two tests complement each other:
- **min-k** is sensitive, catching even a single exposure, but relies on the reference model to cancel out other effects.
- **completion** needs heavier memorisation, but when it fires the evidence is direct. The 10-pass model recalled the exact numbers for 86% of its leaked questions and 5% of the rest.

**Qwen2.5-0.5B** is flagged more often by min-k (16%) than any clean model (at most 10%). It doesn't recall numbers verbatim, though. That's consistent with exposure to GSM8K during training, but it isn't proof: the margin over clean models is small, and Qwen was trained on far more maths data than the reference.

Contamination from a single exposure is hard to detect. That matches the research: [Duan et al., 2024](https://arxiv.org/abs/2402.07841) find membership inference on LLMs is close to chance when text is seen once. Repeated exposure, which is what happens when a benchmark is copied across many web pages, is caught reliably.

## What I learned building it

Every version was checked against clean models (which must pass) and deliberately contaminated ones (which must be caught). Each check found a problem:

1. **The first version flagged GPT-2**, which can't have seen GSM8K. Random replacement numbers read less naturally than numbers a person chose (`8471` vs `150`), so every model preferred the originals. *Fix:* draw replacements from numbers used elsewhere in the benchmark, as often as they're used there, and subtract a reference model's preference.
2. **The second version missed models I had contaminated myself.** Training on a question also made the model more familiar with its twin, because they share 90% of their text, so comparing against "all controls" hid the signal. *Fix:* compare each question with *its own* twin and look only at the difference.
3. **Word-for-word completion barely noticed a model that had memorised its questions.** It writes nearly the same sentence for the twin too. *Fix:* score only the numbers the first half doesn't give away.
4. **A bigger clean model (Pythia-1.4B) was flagged** against the small GPT-2 reference. *Fix:* pick a reference of matching size.

## Limitations

- Twins need numbers. For benchmarks without them, pass `--control` with questions the model can't have seen (for example, written after its training cutoff).
- min-k depends on the reference model cancelling out everything except memorisation. The largest clean reference here is GPT-2 XL (1.5B), so for bigger models pass `--reference` with a larger model trained before the benchmark existed (e.g. `EleutherAI/pythia-6.9b`, trained on 2020 data). Treat a min-k flag on its own as evidence, not proof.
- Validated so far on GSM8K and models up to 1.5B parameters.
- Only open-weight models for now. API models don't expose token probabilities for the prompt.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests use a fake "cheating" model, so they run in under a second without downloading anything.

## License

MIT
