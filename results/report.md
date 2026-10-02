# Dense -> MoE report

- Dense: 4 layers, d_model=128, d_ff=512, 0.82M params
- MoE: 4 experts, top-2, 2.40M total / 1.35M active per token
- Data: char-level, 1,115,394 chars, vocab 65
- Steps: dense 600, MoE 600 (+ dense-continued control)

## Function preservation at conversion (fixed val batches)
dense 1.892581 | MoE noise=0 1.892581 | MoE noise=0.01 1.892677

## Loss-reduction checks

| phase | start val | end val | drop | result |
|---|---|---|---|---|
| dense (from scratch) | 4.2283 | 1.8907 | 2.3376 | PASS |
| MoE (after conversion) | 1.8908 | 1.6590 | 0.2318 | PASS |
| dense continued (control) | 1.8907 | 1.7237 | 0.1671 | PASS |
| MoE vs dense-at-conversion | 1.8907 | 1.6590 | 0.2317 | PASS |

**All checks passed: True**

## Final val loss
- dense @ conversion: 1.8907
- MoE final: 1.6590
- dense continued (control): 1.7237

Plots: loss_curves.png | logs: logs.json | samples: samples.txt
