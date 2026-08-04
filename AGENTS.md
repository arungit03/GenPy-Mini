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
- Phase 3 tokenizer infrastructure and its isolated 1,024-token smoke artifact are complete.
- The production tokenizer remains exactly 16,384 entries and must not be trained or frozen
  until the readiness gate passes on an approved representative corpus.
- Never use the smoke tokenizer for model training or place its fingerprint in model configs.
- Phase 4 may implement the configuration-driven Transformer and token-packing loader, but model
  training remains blocked until the production tokenizer is frozen.
- Phase 4 model, packing, loader, compatibility, and CPU smoke infrastructure is complete.
- Only original safe fixtures may use the smoke tokenizer and smoke model; never pack Phase 2
  production data with them.
- Phase 5 may begin only after the 16,384-token production tokenizer and approved packed training
  corpus are frozen. Scale GenPy-5M before 25M or 100M.
