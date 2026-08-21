"""Deterministic greedy coding evaluation for base and SFT checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genpy.data.io import iter_records, sha256_file
from genpy.evaluation.coding import evaluate_code, generation_summary
from genpy.model import GenPyForCausalLM
from genpy.tokenizer import GenPyTokenizer
from genpy.config import load_config

SANITY_PROMPTS = [
    "Write Python code to check whether a number is even or odd.",
    "Write a Python function to reverse a string.",
    "Write Python code to find the largest number in a list.",
]


def prompt_text(record: dict) -> str:
    instruction = str(record.get("instruction", ""))
    input_text = str(record.get("input", ""))
    return "### User\n" + instruction + (("\n" + input_text) if input_text else "") + "\n\n### Assistant\n"


def generate(model, tokenizer, prompt: str, max_new_tokens: int, device: torch.device) -> tuple[str, bool, int]:
    ids = [tokenizer.bos_token_id, *tokenizer.encode(prompt)]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []
    terminated = False
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids[:, -1024:])[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1).item())
            generated.append(next_id)
            input_ids = torch.cat((input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)), dim=1)
            if next_id == tokenizer.eos_token_id:
                terminated = True
                break
    return tokenizer.decode(generated, skip_special_tokens=True), terminated, len(generated)


def unavailable_report(model_path: Path, dataset_path: Path, output: Path, samples_output: Path) -> int:
    records = [{"prompt": prompt, "status": "BASE_CHECKPOINT_MISSING"} for prompt in SANITY_PROMPTS]
    result = {"status": "BASE_CHECKPOINT_MISSING", "model": str(model_path), "dataset": str(dataset_path), "dataset_sha256": sha256_file(dataset_path) if dataset_path.is_file() else None, "total_prompts": None, "eos_termination_rate": None, "python_extraction_rate": None, "syntax_valid_rate": None, "compile_valid_rate": None, "executable_rate": None, "functional_correctness_rate": None, "repeated_output_rate": None, "unique_output_rate": None, "exact_duplicate_generations": None, "average_generated_tokens": None, "generation_failures": None, "samples": records, "error": "immutable Checkpoint 7 base model is not present in this workspace"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    samples_output.parent.mkdir(parents=True, exist_ok=True)
    samples_output.write_text("Baseline unavailable: immutable Checkpoint 7 model is missing.\n\n" + "\n\n".join(f"PROMPT: {item['prompt']}\nSTATUS: BASE_CHECKPOINT_MISSING" for item in records) + "\n", encoding="utf-8")
    print(f"Baseline unavailable; wrote explicit report to {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/genpy200m_pretrain_v1/checkpoints/step_000000001980/model.pt")
    parser.add_argument("--dataset", default="data/instruction/python/validation.jsonl")
    parser.add_argument("--output", default="reports/checkpoint_8_baseline_eval.json")
    parser.add_argument("--samples-output", default="reports/checkpoint_8_baseline_samples.txt")
    parser.add_argument("--model-config", default="configs/model_200m.yaml")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/genpy-32k")
    parser.add_argument("--max-problems", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    model_path, dataset_path = ROOT / args.model, ROOT / args.dataset
    if dataset_path.name.lower().startswith("test") and not args.final_test:
        raise RuntimeError("refusing test evaluation before explicit --final-test gate")
    if args.final_test:
        manifest_path = ROOT / "data/instruction/tokenized/SFT_TOKEN_CACHE_MANIFEST.json"
        audit_path = ROOT / "reports/checkpoint_8_instruction_data_audit.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
        train_source = str(manifest.get("splits", {}).get("train", {}).get("source", "")).lower()
        validation_source = str(manifest.get("splits", {}).get("validation", {}).get("source", "")).lower()
        if "test" in train_source or "test" in validation_source or not manifest.get("test_split_immutable"):
            raise RuntimeError("final test gate failed: manifest does not prove test exclusion")
        (ROOT / "reports/checkpoint_8_final_test_gate.json").write_text(json.dumps({"test_hash": sha256_file(dataset_path), "test_split_immutable": True, "test_loaded_only_for_final_evaluation": True, "train_test_prompt_overlap": audit.get("cross_split", {}).get("train_test_prompt_overlap"), "train_test_solution_overlap": audit.get("cross_split", {}).get("train_test_solution_overlap")}, indent=2) + "\n", encoding="utf-8")
    if not model_path.is_file():
        return unavailable_report(model_path, dataset_path, ROOT / args.output, ROOT / args.samples_output)
    config = load_config(ROOT / args.model_config)
    model = GenPyForCausalLM(config.model)
    if sum(parameter.numel() for parameter in model.parameters()) != 201560832:
        raise RuntimeError("wrong model parameter count")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
    device = torch.device(args.device)
    model.to(device).eval()
    tokenizer = GenPyTokenizer.load(ROOT / args.tokenizer)
    records = []
    for record in iter_records(dataset_path):
        records.append(record)
        if len(records) >= args.max_problems:
            break
    outputs, evaluations, samples = [], [], []
    failures = 0
    for record in records:
        try:
            generated, eos, token_count = generate(model, tokenizer, prompt_text(record), args.max_new_tokens, device)
            evaluation = evaluate_code(generated, record)
            evaluation.update({"eos": eos, "generated_tokens": token_count})
        except Exception as error:
            failures += 1
            generated, evaluation = "", {"eos": False, "generated_tokens": 0, "failure_category": type(error).__name__, "python_extracted": False, "syntax_valid": False, "compile_valid": False, "executable": None, "functional_correct": None}
        outputs.append(generated); evaluations.append(evaluation)
        if len(samples) < 10:
            samples.append({"instruction": record.get("instruction"), "generated": generated, **evaluation})
    sanity_samples = []
    for prompt in SANITY_PROMPTS:
        generated, eos, token_count = generate(model, tokenizer, prompt_text({"instruction": prompt}), args.max_new_tokens, device)
        sanity_samples.append({"prompt": prompt, "generated": generated, "eos": eos, "generated_tokens": token_count, **evaluate_code(generated)})
    count = len(evaluations)
    rate = lambda key: sum(bool(item[key]) for item in evaluations) / count if count else None
    applicable = [item for item in evaluations if item["functional_correct"] is not None]
    result = {"status": "COMPLETE", "model": str(model_path), "model_sha256": sha256_file(model_path), "dataset": str(dataset_path), "dataset_sha256": sha256_file(dataset_path), "total_prompts": count, "eos_termination_rate": rate("eos"), "python_extraction_rate": rate("python_extracted"), "syntax_valid_rate": rate("syntax_valid"), "compile_valid_rate": rate("compile_valid"), "executable_rate": sum(bool(item["executable"]) for item in applicable) / len(applicable) if applicable else None, "functional_correctness_rate": sum(bool(item["functional_correct"]) for item in applicable) / len(applicable) if applicable else None, "generation_summary": generation_summary(outputs), "average_generated_tokens": sum(item["generated_tokens"] for item in evaluations) / count if count else None, "generation_failures": failures, "samples": samples, "sanity_samples": sanity_samples}
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    samples_output = ROOT / args.samples_output; samples_output.parent.mkdir(parents=True, exist_ok=True); samples_output.write_text(json.dumps({"samples": samples, "sanity_samples": sanity_samples}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "total_prompts", "eos_termination_rate", "syntax_valid_rate", "compile_valid_rate", "functional_correctness_rate", "generation_failures")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
