# Inputs for cookbooks/experiment_analysis.ipynb

Small development artifacts the notebook reads from the repository; everything else comes from the artifacts root.

- `critic_comparison_100q.csv`: the natural-critic validation table (September 2026). The same 100 MuSiQue questions and the same recorded Solver first answers were reviewed by Gemma 3 4B and by Qwen3 4B as Critic under several prompts and token budgets.
- `critic_comparison_example.jsonl`: six records, three of those questions each under the Gemma Critic and the Qwen Critic (one-step prompt, 256-token budget), in Gemma/Qwen alternating order.

Development data from before the production run; not part of the 22,332-episode collection.
