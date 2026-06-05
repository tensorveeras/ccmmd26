import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
import optuna
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoModel, SiglipVisionModel, AutoTokenizer, AutoImageProcessor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gc.collect()

print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    def __init__(self, input_dim, num_experts=2, hidden_dim=256):
        super().__init__()
        self.router = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])

    def forward(self, x):
        gate_probs = F.softmax(self.router(x), dim=-1).unsqueeze(-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return torch.sum(expert_outputs * gate_probs, dim=1)

class CC_MMD_MoE_Model(nn.Module):
    def __init__(self, num_experts=2):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained('microsoft/mdeberta-v3-base')
        self.image_encoder = SiglipVisionModel.from_pretrained('google/siglip-so400m-patch14-384')

        self.text_encoder.gradient_checkpointing_enable()
        self.image_encoder.gradient_checkpointing_enable()

        text_dim = self.text_encoder.config.hidden_size

        if hasattr(self.image_encoder.config, 'vision_config'):
            image_dim = self.image_encoder.config.vision_config.hidden_size
        else:
            image_dim = self.image_encoder.config.hidden_size
            
        combined_dim = text_dim + image_dim

        self.moe = MoELayer(input_dim=combined_dim, num_experts=num_experts, hidden_dim=combined_dim // 2)

        self.head_india = nn.Linear(combined_dim, 1)
        self.head_china = nn.Linear(combined_dim, 1)
        self.head_western = nn.Linear(combined_dim, 1)

    def forward(self, input_ids, attention_mask, pixel_values):
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :] 
        image_features = self.image_encoder(pixel_values=pixel_values).pooler_output

        fused_features = torch.cat((text_features, image_features), dim=1)
        moe_features = self.moe(fused_features)

        final_features = F.relu(moe_features + fused_features)

        return self.head_india(final_features), self.head_china(final_features), self.head_western(final_features)

class CCMMDDataset(Dataset):
    def __init__(self, csv_file, image_dir, text_tokenizer, image_processor, max_length=128):
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.columns = self.data_frame.columns.tolist()

    def __len__(self):
        return len(self.data_frame)

    def _parse_label(self, label_str):
        if pd.isna(label_str):
            return -1.0

        label_str = str(label_str).strip().lower()
        if label_str == 'misogyny':
            return 1.0
        elif label_str == 'not-misogyny':
            return 0.0
        else:
            return -1.0

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]

        img_name = os.path.join(self.image_dir, str(row['image_id']) + '.jpg')
        image = Image.open(img_name).convert('RGB')

        text = str(row['transcriptions']) if pd.notna(row['transcriptions']) else ""

        text_encodings = self.text_tokenizer(
            text, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors="pt"
        )
        image_encodings = self.image_processor(images=image, return_tensors="pt")

        label_india = -1.0
        label_china = -1.0
        label_western = -1.0

        if 'indian_labels' not in self.columns:
            label_india = self._parse_label(row['original_labels'])
            label_china = self._parse_label(row['chinese_labels'])
            label_western = self._parse_label(row['irish_labels'])

        elif 'irish_labels' not in self.columns:
            label_western = self._parse_label(row['original_labels'])
            label_india = self._parse_label(row['indian_labels'])
            label_china = self._parse_label(row['chinese_labels'])

        elif 'chinese_labels' not in self.columns:
            label_china = self._parse_label(row['original_labels'])
            label_india = self._parse_label(row['indian_labels'])
            label_western = self._parse_label(row['irish_labels'])

        return {
            'input_ids': text_encodings['input_ids'].squeeze(0),
            'attention_mask': text_encodings['attention_mask'].squeeze(0),
            'pixel_values': image_encodings['pixel_values'].squeeze(0),
            'labels': torch.tensor([label_india, label_china, label_western], dtype=torch.float16)
        }


tokenizer = AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base')
processor = AutoImageProcessor.from_pretrained('google/siglip-so400m-patch14-384')

def make_dataset(lang, split):
    return CCMMDDataset(
        csv_file=f'./dataset/{lang}/{split}/{split if split == "dev" else "train"}.csv',
        image_dir=f'./dataset/{lang}/{split}/',
        text_tokenizer=tokenizer,
        image_processor=processor
    )

master_train_dataset = ConcatDataset([make_dataset(l, 'train') for l in ['tamil','malayalam','english','chinese']])
master_dev_dataset   = ConcatDataset([make_dataset(l, 'dev')   for l in ['tamil','malayalam','english','chinese']])

train_loader = DataLoader(
    master_train_dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True
)

dev_loader = DataLoader(
    master_dev_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True
)

GRAD_ACCUM_STEPS = 2

def evaluate_ccmmd_model(model, dev_loader, device='cuda'):
    model.eval()
    all_labels_india,   all_preds_india   = [], []
    all_labels_china,   all_preds_china   = [], []
    all_labels_western, all_preds_western = [], []
 
    with torch.no_grad():
        for batch in dev_loader:
            input_ids       = batch['input_ids'].to(device)
            attention_mask  = batch['attention_mask'].to(device)
            pixel_values    = batch['pixel_values'].to(device)
            labels          = batch['labels'].to(device)
 
            logits_india, logits_china, logits_western = model(input_ids, attention_mask, pixel_values)
 
            preds_india   = (torch.sigmoid(logits_india.squeeze(-1))   > 0.5).float()
            preds_china   = (torch.sigmoid(logits_china.squeeze(-1))   > 0.5).float()
            preds_western = (torch.sigmoid(logits_western.squeeze(-1)) > 0.5).float()
 
            y_india, y_china, y_western = labels[:, 0], labels[:, 1], labels[:, 2]
 
            mask_india = (y_india != -1.0)
            all_labels_india.extend(y_india[mask_india].cpu().numpy())
            all_preds_india.extend(preds_india[mask_india].cpu().numpy())
 
            mask_china = (y_china != -1.0)
            all_labels_china.extend(y_china[mask_china].cpu().numpy())
            all_preds_china.extend(preds_china[mask_china].cpu().numpy())
 
            mask_western = (y_western != -1.0)
            all_labels_western.extend(y_western[mask_western].cpu().numpy())
            all_preds_western.extend(preds_western[mask_western].cpu().numpy())
 
    f1_india   = f1_score(all_labels_india,   all_preds_india,   average='macro') if all_labels_india   else 0.0
    f1_china   = f1_score(all_labels_china,   all_preds_china,   average='macro') if all_labels_china   else 0.0
    f1_western = f1_score(all_labels_western, all_preds_western, average='macro') if all_labels_western else 0.0
    final_score = (f1_india + f1_china + f1_western) / 3.0
 
    acc_india   = accuracy_score(all_labels_india,   all_preds_india)   if all_labels_india   else 0.0
    acc_china   = accuracy_score(all_labels_china,   all_preds_china)   if all_labels_china   else 0.0
    acc_western = accuracy_score(all_labels_western, all_preds_western) if all_labels_western else 0.0
 
    print("="*40)
    print(f"India   - macro f1: {f1_india:.4f} | acc: {acc_india:.4f}")
    print(f"China   - macro f1: {f1_china:.4f} | acc: {acc_china:.4f}")
    print(f"Western - macro f1: {f1_western:.4f} | acc: {acc_western:.4f}")
    print(f"task b score: {final_score:.4f}")
    print(f"mean accuracy (tie-breaker): {(acc_india + acc_china + acc_western)/3.0:.4f}")
    print("="*40 + "\n")
    return final_score

def objective(trial, study, train_loader, dev_loader, device='cuda'):
    base_lr           = trial.suggest_float("base_lr", 1e-6, 5e-5, log=True)
    moe_lr_multiplier = trial.suggest_float("moe_lr_multiplier", 1.0, 10.0)
    num_experts       = trial.suggest_categorical("num_experts", [4, 6, 8])
 
    print(f"\n[Worker GPU] Starting trial {trial.number} | lr: {base_lr:.6f} | "
          f"moe multiplier: {moe_lr_multiplier:.1f} | experts: {num_experts}")
 
    torch.cuda.empty_cache()
    model = CC_MMD_MoE_Model(num_experts=num_experts).to(device).float()
 
    optimizer = AdamW([
        {'params': model.text_encoder.parameters(),  'lr': base_lr * 0.1},
        {'params': model.image_encoder.parameters(), 'lr': base_lr * 0.1},
        {'params': model.moe.parameters(),           'lr': base_lr * moe_lr_multiplier},
        {'params': model.head_india.parameters(),    'lr': base_lr * moe_lr_multiplier},
        {'params': model.head_china.parameters(),    'lr': base_lr * moe_lr_multiplier},
        {'params': model.head_western.parameters(),  'lr': base_lr * moe_lr_multiplier},
    ])
 
    num_epochs    = 6
    criterion     = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
 
    total_steps  = (len(train_loader) // GRAD_ACCUM_STEPS) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
 
    best_trial_score = 0.0
 
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
 
        for step, batch in enumerate(train_loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values   = batch['pixel_values'].to(device)
            labels         = batch['labels'].to(device)
 
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits_india, logits_china, logits_western = model(input_ids, attention_mask, pixel_values)
 
                logits_india   = logits_india.squeeze(-1)
                logits_china   = logits_china.squeeze(-1)
                logits_western = logits_western.squeeze(-1)
 
                y_india, y_china, y_western = labels[:, 0], labels[:, 1], labels[:, 2]
 
                valid_losses = []
                valid_india = (y_india != -1.0)
                if valid_india.any():
                    valid_losses.append(criterion(logits_india[valid_india],   y_india[valid_india]))
 
                valid_china = (y_china != -1.0)
                if valid_china.any():
                    valid_losses.append(criterion(logits_china[valid_china],   y_china[valid_china]))
 
                valid_western = (y_western != -1.0)
                if valid_western.any():
                    valid_losses.append(criterion(logits_western[valid_western], y_western[valid_western]))
 
            if valid_losses:
                loss = sum(valid_losses) / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
 
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
 
        val_score = evaluate_ccmmd_model(model, dev_loader, device=device)
 
        if val_score > best_trial_score:
            best_trial_score = val_score
            
            try:
                current_global_best = study.best_value
            except ValueError:
                current_global_best = 0.0

            if val_score > current_global_best:
                print(f"*** New global champion found by Trial {trial.number}! ({current_global_best:.4f} -> {val_score:.4f}) ***")
                print("Overwriting 'ultimate_model.pth' with new best weights...")
                torch.save(model.state_dict(), './ultimate_model_1.pth')
 
        trial.report(val_score, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch} (score too low)")
            raise optuna.exceptions.TrialPruned()
 
    return best_trial_score

if __name__ == "__main__":
    max_seconds = 172800 

    study_name = "ccmmd_sweep"
    storage_name = "sqlite:///optuna_study.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True
    )

    print(f"Worker connected to study '{study_name}'. Starting sweep...")
    study.optimize(
        lambda trial: objective(trial, study, train_loader, dev_loader, device='cuda'), 
        n_trials=25, 
        timeout=max_seconds
    )

    print("="*40)
    try:
        print(f"Best task b score in DB: {study.best_value:.4f}")
        print("Best hyperparameters overall:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
    except ValueError:
        print("No trials completed successfully yet.")
    print("="*40)