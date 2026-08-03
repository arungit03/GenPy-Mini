# Architecture Decisions

This document records initial architecture choices for GenPy. These choices are practical
starting points for a small Python code-generation model, not guarantees of model quality.

## Decoder-Only Transformer

GenPy uses a decoder-only Transformer because the task is autoregressive code generation:
given a prompt, the model predicts the next token until it reaches an end token.

## Causal Self-Attention

Causal self-attention prevents each generated token from attending to future tokens. This
matches the left-to-right generation objective used for code completion and instruction
response generation.

## RoPE

Rotary positional encoding is a common choice for decoder-only models. It avoids learned
absolute position embeddings and is suitable for the planned 1,024-token context length.

## RMSNorm

RMSNorm is computationally simple and commonly used in modern decoder-only models. It is
planned here to keep the architecture efficient and stable during training experiments.

## SwiGLU

SwiGLU feed-forward layers are a standard modern Transformer technique. They may improve
capacity compared with a plain ReLU feed-forward block, but results must be verified.

## Tied Input and Output Embeddings

Tying token input embeddings and output projection weights reduces parameter count and is
reasonable for a compact model.

## 1,024-Token Context

A 1,024-token context is enough for many beginner and intermediate Python tasks while
remaining practical for Kaggle GPU training.

## 16,384-Token Vocabulary

A 16,384-token byte-level BPE vocabulary balances compactness and expressiveness for a
small code model. The vocabulary must be trained only on GenPy data.

## 5M -> 25M -> 100M Scaling

The scaling path allows the pipeline to be tested cheaply before moving to larger runs.
GenPy-5M is for debugging, GenPy-25M is for dataset and scaling validation, and
GenPy-100M is the final target model.
