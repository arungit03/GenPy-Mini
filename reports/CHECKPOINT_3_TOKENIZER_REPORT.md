# GenPy Checkpoint 3 Tokenizer Report

## Status

COMPLETE

## Tokenizer

Name: GenPy-Tokenizer-32K  
Type: Byte-Level BPE  
Vocabulary size: 32000

## Special Tokens

PAD: `<PAD>` / 0  
BOS: `<BOS>` / 1  
EOS: `<EOS>` / 2  
UNK: `<UNK>` / 3

## Dataset

Training file: `data/instruction/python/train.jsonl`  
Training examples: 90,000  
Training file SHA256: `17ba25f0154d1ffa04fdd4b91a22123a0770fe6aa76416ba57e4630264cb0b44`  
Expected SHA256: `17ba25f0154d1ffa04fdd4b91a22123a0770fe6aa76416ba57e4630264cb0b44`  
Hash verification: PASS  
Validation examples: 5,000  
Test examples: 5,000

## Corpus

Characters: 26,903,078  
Bytes: 26,903,078  
Lines: 1,023,741  
Core BPE vocabulary: 13,271  
Corpus-derived completion tokens: 18,729

## Tokenization Metrics

| Split | Characters | Tokens | Tokens/character | Characters/token | UNK tokens | UNK rate |
|---|---:|---:|---:|---:|---:|---:|
| Train | 26,903,078 | 5,578,579 | 0.207358 | 4.8226 | 0 | 0.000000% |
| Validation | 1,493,264 | 309,166 | 0.207040 | 4.8300 | 0 | 0.000000% |
| Test | 1,499,329 | 311,225 | 0.207576 | 4.8175 | 0 | 0.000000% |

## Round-Trip Tests

ASCII: PASS  
Python: PASS  
Multiline Python: PASS  
Indentation: PASS  
Unicode: PASS  
Strings: PASS  
Operators: PASS

## Artifact Validation

Save: PASS  
Reload: PASS  
Encoding stability: PASS  
Vocabulary stability: PASS  
Deterministic smoke rebuild: PASS

## Model Compatibility

Model vocab size: 32000  
Tokenizer vocab size: 32000  
Compatible: YES

## Scope Audit

Pretrained tokenizer used: No  
External pretrained tokenizer files loaded: No  
External vocab reused: No  
External merge table reused: No  
Pretrained model weights used: No  
Transformer implemented: No  
Model training started: No

## Final Result

Checkpoint 3: COMPLETE  
Ready for Checkpoint 4: YES
