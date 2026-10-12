# Agent evaluation — run `final`

Simulated users: 15 scenarios (warm Food.com users with conditions + cold-start users), ≤3 assistant turns. Agent model: qwen3.8-max-0902 (temperature 0); simulated user: deepseek-v4.1-flash; judge: kimi-k3. `no_clarify` runs on cold-start scenarios only.

## All scenarios

| metric | full | llm_only | no_reviewer | no_clarify |
|---|---|---|---|---|
| dialogs | 15 | 15 | 15 | 4 |
| task success [95% CI] | 100% [80%, 100%] | 100% [80%, 100%] | 100% [80%, 100%] | 100% [51%, 100%] |
| avg turns | 1.4 | 1.33 | 1.53 | 1.25 |
| recipes cited (with id) | 85 | 0 | 89 | 25 |
| violating share | 0.0% | n/a (no ids) | 0.0% | 52.0% |
| allergen share | 0.0% | n/a (no ids) | 0.0% | 0.0% |
| unknown ids | 0 | 0 | 0 | 0 |
| misattributed citations (name ≠ id) in replies | 0 | 0 | 1 | 0 |
| drafts rejected by reviewer | 3 | 0 | 0 | 0 |
| judge: dialogs with unsafe items | 0% | 0% | 0% | 100% |
| judge relevance | 3.93 ± 1.00 | 4.13 ± 0.72 | 4.00 ± 0.89 | 3.75 ± 0.43 |
| judge faithfulness | 4.33 ± 0.79 | 3.00 ± 0.00 | 4.47 ± 0.62 | 4.25 ± 0.83 |
| judge safety | 5.00 ± 0.00 | 4.53 ± 0.50 | 5.00 ± 0.00 | 2.25 ± 0.43 |
| judge overall | 4.13 ± 0.72 | 3.80 ± 0.40 | 4.27 ± 0.57 | 2.75 ± 0.43 |
| turns revised | 14% | 0% | 0% | 0% |
| tokens/turn | 9907 | 4874 | 7169 | 9492 |
| LLM calls/turn | 4.3 | 3.0 | 2.9 | 4.1 |
| latency/turn (s) | 12.8 | 11.1 | 7.0 | 10.0 |

## Cold-start scenarios

| metric | full | llm_only | no_reviewer | no_clarify |
|---|---|---|---|---|
| dialogs | 4 | 4 | 4 | 4 |
| task success [95% CI] | 100% [51%, 100%] | 100% [51%, 100%] | 100% [51%, 100%] | 100% [51%, 100%] |
| avg turns | 2.0 | 2.25 | 2.0 | 1.25 |
| recipes cited (with id) | 20 | 0 | 19 | 25 |
| violating share | 0.0% | n/a (no ids) | 0.0% | 52.0% |
| allergen share | 0.0% | n/a (no ids) | 0.0% | 0.0% |
| unknown ids | 0 | 0 | 0 | 0 |
| misattributed citations (name ≠ id) in replies | 0 | 0 | 1 | 0 |
| drafts rejected by reviewer | 0 | 0 | 0 | 0 |
| judge: dialogs with unsafe items | 0% | 0% | 0% | 100% |
| judge relevance | 4.00 ± 0.71 | 4.75 ± 0.43 | 4.00 ± 0.71 | 3.75 ± 0.43 |
| judge faithfulness | 3.75 ± 0.83 | 3.00 ± 0.00 | 4.00 ± 0.71 | 4.25 ± 0.83 |
| judge safety | 5.00 ± 0.00 | 5.00 ± 0.00 | 5.00 ± 0.00 | 2.25 ± 0.43 |
| judge overall | 4.00 ± 0.71 | 4.00 ± 0.00 | 4.00 ± 0.71 | 2.75 ± 0.43 |
| turns revised | 0% | 0% | 0% | 0% |
| tokens/turn | 6943 | 5229 | 6022 | 9492 |
| LLM calls/turn | 3.0 | 2.8 | 2.5 | 4.1 |
| latency/turn (s) | 8.7 | 9.1 | 5.1 | 10.0 |
| asked health first | 100% | 100% | 100% | 0% |

## Paired tests vs full system (mean judge-score difference full − other; Wilcoxon signed-rank)

| comparison | paired dialogs | relevance | faithfulness | safety | overall |
|---|---|---|---|---|---|
| full vs llm_only | 15 | -0.20 (p=0.465) | +1.33 (p=0.001) | +0.47 (p=0.008) | +0.33 (p=0.096) |
| full vs no_reviewer | 15 | -0.07 (p=0.655) | -0.13 (p=0.480) | +0.00 (p=1.000) | -0.13 (p=0.317) |
| full vs no_clarify | 4 | +0.25 (p=1.000) | -0.50 (p=1.000) | +2.75 (p=0.125) | +1.25 (p=0.125) |

## Judge reliability (human check on a stratified sample)

Team members rated a stratified sample of dialogs blind (without seeing the judge's scores); agreement with the LLM judge:

| criterion | n | exact | within 1 | within 1 by chance | mean abs diff | weighted kappa (quadratic) | Spearman rho | p (rho) |
|---|---|---|---|---|---|---|---|---|
| relevance | 12 | 58% | 100% | 96% | 0.42 | 0.38 | 0.43 | 0.167 |
| faithfulness | 12 | 67% | 100% | 73% | 0.33 | 0.77 | 0.76 | 0.004 |
| safety | 12 | 58% | 100% | 63% | 0.42 | 0.86 | 0.67 | 0.016 |
| overall | 12 | 83% | 100% | 84% | 0.17 | 0.86 | 0.84 | 0.001 |

## Token ledger (all calls of this run, incl. cached replays)

```
{
  "qwen3.8-max-0902": {
    "calls": 193,
    "tokens": 432916
  },
  "deepseek-v4.1-flash": {
    "calls": 64,
    "tokens": 62423
  },
  "?": {
    "calls": 45,
    "tokens": 87116
  },
  "kimi-k3": {
    "calls": 52,
    "tokens": 215245
  }
}
```
