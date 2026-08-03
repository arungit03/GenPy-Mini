# GenPy Agent Rules

These rules are permanent project constraints for any future coding agent working on GenPy.

- GenPy is a Python code-generation language model trained from random weights.
- Existing LLM weights and tokenizers are forbidden, including GPT, Qwen, Llama,
  Claude, StarCoder, and similar systems.
- The final target model size is approximately 100 million parameters.
- Development must follow the scaling path: GenPy-5M -> GenPy-25M -> GenPy-100M.
- Code must be configuration-driven. Avoid hard-coded training, model, or path settings.
- Every major component requires tests before it is considered complete.
- Paths must work on Windows and Linux. Use `pathlib` in Python code.
- Dataset licences and sources must be recorded before data is used.
- Evaluation data must remain separate from training data.
- Generated code must not be executed without an isolated sandbox with network, time,
  memory, and filesystem restrictions.
- Existing working files must not be overwritten carelessly.
- Do not commit secrets, datasets, checkpoints, generated artifacts, or large model files.
- Phase boundaries matter. Do not collect datasets, train tokenizers, implement the full
  Transformer, or start training during Phase 1.
