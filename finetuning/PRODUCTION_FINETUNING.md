# Production Fine-tuning - Setup Guide

## Übersicht

Vorbereitet für Production Fine-tuning mit **allen 16 SMI-Aktien** und optimierten Hyperparametern.

## Files

- `finetune_production.py` - Main training script
- `compare_production_models.py` - Evaluation script
- **Noch NICHT gestartet** - Manueller Start erforderlich

## Optimierte Parameter

```python
# Deutlich verbessert vs. Quick Test (5 Aktien, 500 Steps)
num_steps = 5000          # 10x mehr Training
learning_rate = 1e-6      # Stabilere Fine-tuning
batch_size = 16           # Alle Aktien = Cross-Learning
prediction_length = 5     # Längerer Horizont
```

## Expected Performance

**Basierend auf Literatur:**
- Mit 16 Aktien: +10-20% Improvement möglich
- Mit mehr Steps: Bessere Konvergenz
- Mit Cross-Learning: Pattern-Sharing zwischen Aktien

## Training Time

**GPU (RTX 3050):**
- ~60 Minuten geschätzt
- 4.2 steps/sec @ 5000 steps

**CPU:**
- ~75 Stunden (NICHT empfohlen!)

## How to Run

### 1. Starte Training (manuell)

```bash
python finetune_production.py
```

→ Fragt nach Bestätigung bevor Start!

### 2. Nach Training: Vergleich

```bash
python compare_production_models.py
```

→ Testet auf allen 16 Aktien

## Output

```
production_finetuning/
├── model/                    # Fine-tuned model
├── training_config.json      # Training parameters
└── data_metadata.json        # Data statistics

production_evaluation/
└── comparison.json           # Baseline vs Fine-tuned results
```

## Deployment Decision

Script gibt automatisch Empfehlung:

- **>10% Improvement:** ✅ Deploy Fine-tuned
- **5-10%:** ⚠️  Consider
- **<5%:** Use Baseline

## Monitoring

Training zeigt alle 100 Steps:
- Training Loss
- Validation Loss (if provided)
- Steps/sec
- ETA

## Notes

- GPU muss verfügbar sein (PyTorch CUDA)
- Alle 16 SMI-Aktien werden geladen
- Daten automatisch aus Cache
- Cross-Learning aktiviert durch batch_size=16

## Troubleshooting

**GPU nicht erkannt:**
```bash
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**Speicher-Probleme:**
- Reduziere `batch_size` auf 8
- Oder nutze `prediction_length=1`

**Zu langsam:**
- Checke GPU-Nutzung: `nvidia-smi`
- Stelle sicher CUDA funktioniert
