# Example TinyPacketCNN checkpoints

Trained with `config_classification_drone` (TinyPacketCNN, `input_size=5`,
`output_size=1`, ~14k params) on the SITL packet classification workload.

| File | Defender | Attack | Seed |
|------|----------|--------|------|
| `byzantine_seed42_acfl_full.pt` | ACFL (full) | Byzantine | 42 |
| `byzantine_seed42_fedavg.pt` | FedAvg | Byzantine | 42 |

```bash
python -c "
import sys, torch
sys.path.insert(0, 'fl-time-series')
from models.tiny_packet_cnn import TinyPacketCNN
ckpt = torch.load(
    'fl-time-series/checkpoints/acfl_examples/byzantine_seed42_acfl_full.pt',
    map_location='cpu', weights_only=False,
)
m = TinyPacketCNN(
    input_size=ckpt['input_size'],
    output_size=ckpt['output_size'],
)
m.load_state_dict(ckpt['model_state_dict'])
print('round', ckpt['round'], 'params', m.count_parameters())
"
```
