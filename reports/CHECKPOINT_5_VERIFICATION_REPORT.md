# GenPy Checkpoint 5 Verification Report

## Status

LOCAL_COMPLETE_GPU_PENDING

## Model

Model: GenPy-200M  
Parameters: 201,560,832  
Layers: 24  
Hidden: 768  
Heads: 12  
FFN: 2176  
Vocabulary: 32000  
Context: 1024

## Core Verification

Forward: PASS  
Backward: PASS  
Gradient finiteness: PASS  
Loss correctness: PASS (difference 0)  
Causal isolation: PASS  
RoPE: PASS  
RMSNorm: PASS  
Weight tying: PASS  
Weight tying after reload: PASS  
Save/load numerical stability: PASS  
NaN/Inf checks: PASS

## Tiny Overfit

Steps: 200  
Initial loss: 4.883588  
Final loss: 0.004108  
Loss reduction: 99.92%  
Result: PASS

## GPU

Device: CPU / no CUDA  
CUDA: NO  
BF16 supported: NO

FP32: SKIPPED_NO_CUDA  
Mixed precision: SKIPPED_NO_CUDA  
Full context 1024: SKIPPED_NO_CUDA

## Scope Audit

Production training started: No  
Instruction tuning started: No  
Training engine implemented: No

## Tests

Passed: 59  
Failed: 0  
Skipped: GPU/1024-context checks pending CUDA hardware

## Final Result

Checkpoint 5: LOCAL_COMPLETE_GPU_PENDING  
Ready for Checkpoint 6: NO
