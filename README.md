# autoresearch

Autonomous ML research platform where AI agents explore model configurations, train experiments, and discover better architectures without human intervention.

This project applies the [autoresearch](https://github.com/karpathy/autoresearch) pattern to visual re-identification (ReID): a lightweight student model (LCNet) is distilled from multiple large teacher models (DINOv2, DINOv3, C-RADIO) through autonomous experimentation.

## Project Structure

Each research topic lives in its own subfolder, designed to be run with that directory as the working directory.

```
autoresearch/
  student_finetune/     Student model distillation (LCNet) from multiple teachers
    prepare.py          Data loading, teacher definitions, TEACHER_REGISTRY
    train.py            Training loop, model architecture, loss functions
    build_caches.py     Pre-build teacher embedding caches
    tests/              Test suite for student distillation code
  dino_finetune/        DINOv3 ViT-H+ fine-tuning with LoRA for teacher model
    train_dino.py       DINOv3 contrastive fine-tuning with LoRA adapters
  RADIO/                Local clone of C-RADIO model repository (torch.hub source)
  workspace/            Runtime artifacts (teacher caches, outputs, results)
  program.md            Agent instructions for autonomous experimentation
  pyproject.toml        Project dependencies
```

## Usage

### Student Distillation

```bash
cd student_finetune

# Build teacher embedding caches (one-time, requires GPU)
python build_caches.py

# Run training
python train.py
```

### DINOv3 Fine-tuning

```bash
cd dino_finetune
python train_dino.py
```

### Running Tests

```bash
cd student_finetune
python -m pytest tests/ -x -q
```

## Adding New Research Topics

Create a new subfolder following the existing pattern:

1. Create `new_topic/` with its own `train.py` and supporting modules
2. Use relative paths with `../` prefix to reference shared resources (e.g., `../workspace/`, `../RADIO/`)
3. Each subfolder should be self-contained and runnable from its own directory as CWD
