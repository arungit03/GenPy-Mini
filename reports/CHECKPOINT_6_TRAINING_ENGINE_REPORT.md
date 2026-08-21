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
BOS/EOS audit: PASS
Cache integrity: PASS

## Engine Verification

Memmap dataset: PASS
Causal shift: PASS
AdamW and parameter coverage: PASS
Gradient accumulation/clipping/non-finite guards: PASS
Warmup/cosine scheduler and resume: PASS
Deterministic validation: PASS
Atomic checkpoints and RNG restore: PASS
Exact local resume: PASS

## GPU Smoke Evidence

Evidence file: `C:\Users\arun0\OneDrive\Desktop\GenPy-Mini\reports\checkpoint_6_gpu_report.json`
Production CUDA/BF16 smoke: PENDING

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
