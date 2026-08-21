# GenPy Checkpoint 6 Training Engine Report

## Status

LOCAL_COMPLETE_GPU_SMOKE_PENDING

## Token Cache

Train documents: 90,000  
Train tokens: 5,398,579  
Validation documents: 5,000  
Validation tokens: 299,166  
Storage dtype: uint16  
Train hash: `17ba25f0154d1ffa04fdd4b91a22123a0770fe6aa76416ba57e4630264cb0b44`  
Validation hash: `7ec5fabfb339e0c9986160193da01d30fe6d46035775f81e9777a4ad92731e97`  
BOS count: 90,000 train / 5,000 validation  
EOS count: 90,000 train / 5,000 validation  
BOS/EOS audit: PASS  
Cache integrity: PASS

## Dataset

Memmap: PASS  
Causal x/y shift: PASS

## Optimizer

Type: AdamW  
Learning rate: 0.0003  
Betas: 0.9 / 0.95  
Epsilon: 1e-8  
Weight decay: 0.1  
Parameter coverage: PASS  
Duplicate params: 0  
Missing params: 0

## Scheduler

Warmup: PASS  
Cosine decay: PASS  
Minimum LR: PASS  
Resume: PASS

## Gradient Accumulation

Micro batch: 1  
Sequence: 1024 production default  
Accumulation: 8 production default  
Effective tokens/update: 8192  
Result: PASS

## Mixed Precision

Device: CPU  
Mode: FP32  
GradScaler: unused for FP32  
Result: FP32 PASS; BF16/FP16 CUDA paths implemented and pending GPU smoke

## Validation

Validation loop: PASS  
Deterministic: PASS

## Checkpoints

Atomic save: PASS  
Latest pointer: PASS  
Optimizer restore: PASS  
Scheduler restore: PASS  
RNG restore: PASS

## Exact Resume

Uninterrupted steps: 6  
Interrupted step: 3  
Resumed final step: 6  
Model equality: PASS  
Optimizer equality: PASS  
Scheduler equality: PASS  
Batch RNG equality: PASS  
Loss continuation: PASS  
Result: PASS

## Smoke Training

Model: tiny verification model  
Steps: 6  
Initial loss: 4.901859  
Final loss: 4.850522  
Tokens: 192  
Checkpoint save: PASS  
Exact resume: PASS

## Tests

Passed: 74  
Failed: 0  
Skipped: CUDA production smoke pending

## Scope Audit

Production pretraining started: No  
Instruction tuning started: No  
Multi-GPU implemented: No

## Final Result

Checkpoint 6: LOCAL_COMPLETE_GPU_SMOKE_PENDING  
Ready for Checkpoint 7: NO
