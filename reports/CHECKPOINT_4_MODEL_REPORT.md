# GenPy Checkpoint 4 Model Report

## Status

COMPLETE

## Architecture

Model: GenPy-200M  
Architecture: Decoder-only Transformer  
Layers: 24  
Hidden size: 768  
Heads: 12  
Head dimension: 64  
FFN hidden size: 2176  
Vocabulary: 32000  
Context: 1024

## Components

RMSNorm: PASS  
RoPE: PASS  
Causal Multi-Head Attention: PASS  
SwiGLU: PASS  
Residual Architecture: PASS  
Final RMSNorm: PASS  
Weight Tying: PASS  
Bias Audit: PASS

## Parameter Count

Embedding: 24,576,000  
Attention: 56,623,104  
SwiGLU: 120,324,096  
Block norms: 36,864  
Final norm: 768  
Expected: 201,560,832  
Actual: 201,560,832  
Parameter count: PASS

## Tokenizer Compatibility

Tokenizer vocabulary: 32000  
Model vocabulary: 32000  
PAD/BOS/EOS/UNK: 0/1/2/3  
Compatibility: PASS

## Forward Tests

Tiny forward: PASS  
Output shape: PASS  
Causal isolation: PASS  
Maximum context guard: PASS  
Gradient smoke: PASS  
Numerical finiteness: PASS

## Initialization

Random initialization: PASS  
Reproducibility: PASS  
Pretrained model weights: No

## Tests

Passed: 50  
Failed: 0

## Scope Audit

Training engine implemented: No  
Production training started: No  
Instruction tuning started: No

## Final Result

Checkpoint 4: COMPLETE  
Ready for Checkpoint 5: YES
