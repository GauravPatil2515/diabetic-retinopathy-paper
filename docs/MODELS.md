# Models Architecture Documentation

## Overview

This directory contains all model architectures and pretrained weights for the Diabetic Retinopathy classification system.

---

## Architecture: Hybrid CNN-ViT with Domain-Specific Batch Normalization

### Main Model: `HybridDINOv2_DSBN` (hybrid_v3.py)

#### Purpose
Combines multi-scale CNN features with global ViT context via deep cross-attention, where all normalization layers are **domain-aware** (separate statistics per dataset).

#### Architecture Diagram
```
Input: [B, 3, 224, 224]
  │
  ├─ EfficientNet-B3 (features_only)
  │  └─ Multi-scale outputs: C3, C4, C5 (stage 3,4,5)
  │     └─ Shapes: [B, 192, 56, 56], [B, 240, 28, 28], [B, 384, 14, 14]
  │
  └─ DINOv2 ViT-Small (forward_features)
     └─ Global context: [B, 384]

  ↓ Projection & Domain-Specific Norm
  
  for i in [0, 1, 2]:  # Multi-scale fusion
    CNN_feat ← Conv(1×1) → DSBatchNorm2d → Sequence
    ViT_feat ← Linear → DSLayerNorm → Query
    Fused ← DeepCrossAttention(CNN, ViT, domain_labels)
  
  ↓ Final Classification
  
  Pooled = LayerNorm(Fused)
  Output = Linear(256 → 5)
```

#### Key Components

**1. Domain-Specific Batch Normalization (DSBatchNorm2d)**
```python
class DSBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_domains=2):
        super().__init__()
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(num_features) for _ in range(num_domains)
        ])
    
    def forward(self, x, domain_labels):
        # Route to domain-specific BN
        for d in range(self.num_domains):
            mask = (domain_labels == d)
            x[mask] = self.bns[d](x[mask])
        return x
```

**Why Domain Routing?**
- APTOS images: high resolution, professional acquisition
  - → Tighter activation distributions
  - → Different optimal BN running_mean/var
- Messidor images: variable compression, older phones
  - → Broader activation ranges
  - → Separate batch statistics prevent domain confusion

**2. Domain-Specific Layer Norm (DSLayerNorm)**
```python
class DSLayerNorm(nn.Module):
    def __init__(self, normalized_shape, num_domains=2):
        self.lns = nn.ModuleList([
            nn.LayerNorm(normalized_shape) for _ in range(num_domains)
        ])
    
    def forward(self, x, domain_labels):
        for d in range(self.num_domains):
            mask = (domain_labels == d)
            x[mask] = self.lns[d](x[mask])
        return x
```

**3. Deep Cross-Attention Block**
```python
class DeepCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4, num_domains=2):
        self.norm1 = DSLayerNorm(hidden_dim, num_domains)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm2 = DSLayerNorm(hidden_dim, num_domains)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, query, key_value, domain_labels):
        # Normalized cross-attention
        q = self.norm1(query, domain_labels)
        kv = self.norm1(key_value, domain_labels)
        attn_out, _ = self.attn(q, kv, kv)
        query = query + attn_out
        
        # FFN with domain-aware normalization
        query = query + self.ffn(self.norm2(query, domain_labels))
        return query
```

#### Forward Pass Details

```python
def forward(self, images, domain_labels, alpha=0.0):
    # alpha = gradient reversal strength (for adversarial training)
    
    # 1. Feature Extraction
    cnn_features = self.cnn(images)  # List of 3 tensors
    vit_features = self.vit.forward_features(images)
    vit_cls = vit_features['x_norm'][:, 0]  # CLS token
    
    # 2. Projection & Normalization
    vit_proj = self.vit_proj(vit_cls)  # [B, 256]
    vit_proj = self.vit_dsln(vit_proj, domain_labels)  # Domain-aware norm
    
    query = vit_proj.unsqueeze(1)  # [B, 1, 256]
    
    # 3. Multi-Scale Cross-Attention
    for i, (feat, proj, dsbn, block) in enumerate(
        zip(cnn_features, self.cnn_projs, self.cnn_dsbns, self.cross_attn_blocks)
    ):
        # Project CNN feature
        proj_feat = proj(feat)  # [B, 256, H, W]
        proj_feat = dsbn(proj_feat, domain_labels)  # Domain routing
        
        # Flatten to sequence
        H, W = proj_feat.shape[2:]
        kv_seq = proj_feat.flatten(2).transpose(1, 2)  # [B, H*W, 256]
        
        # Cross attention
        query = block(query, kv_seq, domain_labels)
    
    # 4. Classification
    fused = query.squeeze(1)  # [B, 256]
    fused = self.final_dsln(fused, domain_labels)
    dr_logits = self.dr_classifier(fused)  # [B, 5]
    
    return dr_logits, domain_logits
```

---

## Ablation Model: `HybridDINOv2_SharedNorm` (hybrid_v4_shared_norm.py)

### Purpose
Validates the necessity of DSBN by replacing all domain-aware normalization with standard shared norms.

### Key Differences from DSBN
```python
# DSBN:
self.cnn_dsbns = nn.ModuleList([
    DSBatchNorm2d(hidden_dim, num_domains=2)
])

# SharedNorm:
self.cnn_bns = nn.ModuleList([
    nn.BatchNorm2d(hidden_dim)  # Single BN for all domains
])

# DSBN:
self.vit_dsln = DSLayerNorm(hidden_dim, num_domains=2)

# SharedNorm:
self.vit_ln = nn.LayerNorm(hidden_dim)  # Single LN
```

### Results (from Experiment 2)
```
DSBN:          Overall QWK = 0.8437 ✅
SharedNorm:    Overall QWK = 0.7422 ❌ (-12.0%)
```

**Conclusion**: Domain-aware routing is essential, not optional preprocessing.

---

## Pretrained Weights Structure

### Directory Layout
```
project/models/pretrained/
├── densenet121/
│   └── densenet121-a639ec97.pth
├── dino_vits16/
│   └── dino_deitsmall16_pretrain.pth
├── efficientnet_b3/
│   └── efficientnet_b3_rwightman-b3899882.pth
└── vit_b16/
    └── vit_b_16-c867db91.pth
```

### Loading Pretrained Weights
```python
from project.models.hybrid_v3 import HybridDINOv2_DSBN

pretrained_dir = 'project/models/pretrained'
model = HybridDINOv2_DSBN(
    cnn_name='efficientnet_b3',
    vit_name='vit_small_patch16_224',
    num_classes=5,
    num_domains=2,
    pretrained_dir=pretrained_dir  # Loads local weights
)
```

---

## Checkpoints: Trained Models

### Baseline Checkpoint
```
Name: hybrid_dann_efficientnet_b3_vit_small_patch16_224_best.pth
Size: 132 MB
QWK:  0.8437 (Overall), 0.8872 (APTOS), 0.6643 (Messidor)
Epoch: 20/30

Contents:
  - model_state_dict: All weights & biases
  - optimizer_state_dict: SAM optimizer state
  - scheduler_state_dict: ReduceLROnPlateau state
  - best_qwk: Best validation score tracked
  - args: Training hyperparameters
```

### Loading Checkpoint
```python
import torch
from project.models.hybrid_v3 import HybridDINOv2_DSBN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build model
model = HybridDINOv2_DSBN(
    cnn_name='efficientnet_b3',
    vit_name='vit_small_patch16_224',
    num_classes=5,
    num_domains=2
).to(device)

# Load checkpoint
checkpoint = torch.load(
    'project/models/checkpoints/hybrid_dann_efficientnet_b3_vit_small_patch16_224_best.pth',
    map_location=device
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    images = torch.randn(1, 3, 224, 224).to(device)
    domain_labels = torch.tensor([0]).to(device)  # 0=APTOS, 1=Messidor
    logits, _ = model(images, domain_labels, alpha=0.0)
    predictions = torch.argmax(logits, dim=1)
```

---

## Model Selection Guide

| Use Case | Model | Config |
|----------|-------|--------|
| **Production** | DSBN | `model_type='dsbn'` (default) |
| **Ablation Study** | SharedNorm | `model_type='shared_norm'` |
| **Research** | DSBN + EMA | `enable_ema=True` |
| **Inference** | DSBN (best ckpt) | Load from checkpoints/ |

---

## Hyperparameter Reference

### Model Architecture
| Parameter | Value | Notes |
|-----------|-------|-------|
| CNN Backbone | EfficientNet-B3 | Pre-trained on ImageNet |
| ViT Backbone | DINOv2 ViT-Small | Pre-trained on unlabeled data |
| Hidden Dimension | 256 | Projection target dimension |
| Attention Heads | 4 | Multi-head attention |
| Num Cross-Attention Blocks | 3 | One per CNN scale |
| Classification Head | Linear(256→5) | Severity classes: 0-4 |

### Training
| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning Rate | 2e-5 | Conservative for ViT fine-tuning |
| Batch Size | 16 | Balance memory & gradients |
| Optimizer | SAM + AdamW | Sharpness awareness |
| Scheduler | ReduceLROnPlateau | Adaptive LR reduction |
| Loss | DistanceWeightedCE | Ordinal penalty |

---

## Common Issues & Solutions

### Issue: Model loads but predictions are garbage
**Solution**: Ensure domain_labels are provided
```python
# ❌ Wrong
logits, _ = model(images)  # domain_labels missing!

# ✅ Correct
domain_labels = torch.zeros(batch_size, dtype=torch.long).to(device)
logits, _ = model(images, domain_labels, alpha=0.0)
```

### Issue: Memory overflow during training
**Solution**: Reduce batch size or use gradient accumulation
```bash
python train_domain_aware.py --batch_size 8 --lr 2e-5
```

### Issue: DSBN randomly produces NaN
**Solution**: Check domain_labels shape and values
```python
assert domain_labels.shape == (batch_size,)
assert domain_labels.min() >= 0 and domain_labels.max() <= 1
```

---

**Last Updated**: April 9, 2026  
**Maintainer**: Gaurav Patil
