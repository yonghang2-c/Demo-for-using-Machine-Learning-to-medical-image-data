import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from skimage.measure import label as sk_label, regionprops
from tqdm import tqdm
import shutil
import optuna
import json
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from joblib import dump
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from math import sqrt

# ------------------
# 1. Reproducibility
# ------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------
# 2. Transform (pickle-safe)
# ------------------
class SegmentationTransform:
    def __init__(self):
        self.img_tf = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        self.mask_tf = T.ToTensor()
    def __call__(self, image, mask):
        return self.img_tf(image), self.mask_tf(mask)

# ------------------
# 3. Dataset
# ------------------
class ChestXrayDataset(Dataset):
    def __init__(self, df_bbox, image_dir, transform=None):
        self.df_bbox   = df_bbox
        self.ids       = df_bbox['ImageIndex'].unique().tolist()
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert('L')
        w, h = image.size
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for _, r in self.df_bbox[self.df_bbox['ImageIndex']==img_id].iterrows():
            x, y, bw, bh = map(int, (
                r['x'], r['y'], r['width'], r['height']
            ))
            draw.rectangle([x, y, x+bw, y+bh], fill=1)
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask, img_id

def main():
    set_seed(42)
    # ------------------
    # 4. Paths & Params
    # ------------------

    DATA_DIR = "/Users/laiyonghang/PycharmProjects/PythonProject1/demo"
    IMG_DIR = os.path.join(DATA_DIR, "images")

    os.makedirs(IMG_DIR, exist_ok=True)

    # 2) 遍历 images1…images12，把图片复制（或链接）到 images/
    for i in range(1, 13):
        src_dir = os.path.join(DATA_DIR, f"images{i}")
        if not os.path.isdir(src_dir):
            continue
        for fname in os.listdir(src_dir):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(IMG_DIR, fname)
            if os.path.exists(dst_path):
                continue
            # 如果想要节省空间，用软链接：
            # os.symlink(src_path, dst_path)
            # 如果想要真复制一份，就用下面这一行：
            shutil.copy(src_path, dst_path)

    IMG_DIR = os.path.join(DATA_DIR, "images")
    META_PATH = os.path.join(DATA_DIR, "Data_Entry_2017_v2020.csv")
    BBOX_PATH = os.path.join(DATA_DIR, "BBox_List_2017.csv")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 8
    VAL_SPLIT  = 0.2
    NUM_EPOCHS = 20
    LR         = 1e-3
    PATIENCE   = 5
    NUM_WORKERS= 2

    print("DATA_DIR  =", DATA_DIR)
    print("IMG_DIR   =", IMG_DIR)
    print("META_PATH =", META_PATH)
    print("BBOX_PATH =", BBOX_PATH)
    print("DEVICE    =", DEVICE)

    # ------------------
    # 5. Load & Rename Columns
    # ------------------
    meta = pd.read_csv(META_PATH)
    meta = meta.rename(columns={
        'Image Index': 'ImageIndex',
        'Patient ID': 'PatientID',
        'Finding Labels': 'Label',
        'Patient Age': 'PatientAge',
        'Patient Gender': 'PatientGender',
        'View Position': 'ViewPosition',
        'Follow-up #': 'FollowUp',
        'OriginalImage[Width': 'width',
        'Height]': 'height',
        'OriginalImagePixelSpacing[x': 'x',
        'y]': 'y'
    })

    print(meta.columns.tolist())

    bbox = pd.read_csv(BBOX_PATH)
    bbox = bbox.rename(columns={
        'Image Index': 'ImageIndex',
        'Finding Label': 'Label',
        'Bbox [x': 'x',
        'h]': 'height',
        'w': 'width'
    })

    bbox = bbox[['ImageIndex','Label','x','y','width','height']]
    print(bbox.columns.tolist())

    existing_imgs = set(os.listdir(IMG_DIR))
    meta = meta[meta['ImageIndex'].isin(existing_imgs)]
    bbox = bbox[bbox['ImageIndex'].isin(existing_imgs)]

    # only Pneumonia
    pneu = bbox[bbox['Label']=="Pneumonia"]
    ids  = pneu['ImageIndex'].unique().tolist()

    # ------------------
    # 6. DataLoaders
    # ------------------
    transform = SegmentationTransform()
    train_ids, val_ids = train_test_split(ids, test_size=VAL_SPLIT, random_state=42)
    train_bbox = pneu[pneu['ImageIndex'].isin(train_ids)]
    val_bbox   = pneu[pneu['ImageIndex'].isin(val_ids)]

    train_ds = ChestXrayDataset(train_bbox, IMG_DIR, transform=transform)
    val_ds   = ChestXrayDataset(val_bbox,   IMG_DIR, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    # ------------------
    # 7. Model, Loss, Optimizer
    # ------------------
    model     = smp.Unet("resnet34", encoder_weights="imagenet",
                         in_channels=1, classes=1).to(DEVICE)
    bce_loss  = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(mode='binary')
    def loss_fn(logits, masks):
        return bce_loss(logits, masks) + dice_loss(logits, masks)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    scaler    = GradScaler()

    # ------------------
    # 8. Evaluation
    # ------------------
    def evaluate(loader):
        model.eval()
        losses, dices = [], []
        with torch.no_grad():
            for imgs, masks, _ in loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                with autocast(device_type=DEVICE.type):
                    logits = model(imgs)
                    loss   = loss_fn(logits, masks)
                losses.append(loss.item())
                preds = (torch.sigmoid(logits)>0.5).float()
                inter = (preds * masks).sum(dim=(1,2,3))
                union = preds.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3))
                dices.append(((2*inter+1e-7)/(union+1e-7)).mean().item())
        return np.mean(losses), np.mean(dices)

    # ------------------
    # 9. Training Loop
    # ------------------
    history = []

    best_val_loss = float('inf')
    no_improve    = 0
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        train_losses = []
        for imgs, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type=DEVICE.type):
                logits = model(imgs)
                loss   = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        val_loss, val_dice = evaluate(val_loader)
        scheduler.step(val_loss)

        history.append({
            'epoch': epoch,
            'train_loss': float(np.mean(train_losses)),
            'val_loss':    float(val_loss),
            'val_dice':    float(val_dice),
        })

        print(f"Train Loss: {history[-1]['train_loss']:.4f} | "
              f"Val Loss: {history[-1]['val_loss']:.4f} | "
              f"Val Dice: {history[-1]['val_dice']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), "best_unet.pth")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping...")
                break

    pd.DataFrame(history).to_csv("training_history.csv", index=False)
    print("Saved epoch history to training_history.csv")

    # ------------------
    # 10. Inference & Feature Extraction
    # ------------------
    model.load_state_dict(torch.load("best_unet.pth", weights_only=True))
    model.eval()
    feats = []
    with torch.no_grad():
        for imgs, _, img_ids in tqdm(val_loader, desc="Extracting Features"):
            imgs = imgs.to(DEVICE)
            logits = model(imgs)  # 直接用 float32
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs>0.5).astype(np.uint8)
            for i, mask in enumerate(preds[:,0]):
                total = mask.size
                area  = mask.sum()
                labeled = sk_label(mask)
                props   = regionprops(labeled)
                areas     = [p.area for p in props]
                centroids = [p.centroid for p in props]
                dists     = [np.linalg.norm(np.array(c)-np.array(mask.shape)/2) for c in centroids]
                feats.append({
                    'ImageIndex':       img_ids[i],
                    'LesionRatio':      area/total,
                    'NumLesions':       len(areas),
                    'MeanArea':         np.mean(areas) if areas else 0,
                    'MeanCentroidDist': np.mean(dists) if dists else 0
                })
    sp_df = pd.DataFrame(feats)
    sp_df.to_csv("spatial_features.csv", index=False)
    print("Saved spatial features to spatial_features.csv")

    # ------------------
    # 11. Merge & Regression
    # ------------------
    sp_df['PatientID'] = sp_df['ImageIndex'].str.split('_').str[0]
    clin = meta[['PatientID','PatientAge','PatientGender']].drop_duplicates()
    clin['PatientID'] = clin['PatientID'].astype(int).astype(str).str.zfill(8)
    clin['SexEncoded'] = clin['PatientGender'].map({'M':0,'F':1})

    data = sp_df.merge(clin, on='PatientID', how='inner').dropna()
    if data.empty:
        raise RuntimeError("No samples，Please PatientID")

    X = data[['PatientAge', 'SexEncoded', 'NumLesions', 'MeanArea', 'MeanCentroidDist']]
    y = data['LesionRatio']

    RANDOM_SEED = 42

    def objective(trial, model_type, X, y):
        # 根据 model_type 构建不同的超参空间
        if model_type == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 6, 16),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1e1, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1e1, log=True),
                'random_state': 42,
                'n_jobs': 1,
            }
            model = XGBRegressor(**params)

        elif model_type == 'lgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 255, step=31),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50, step=10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1e1, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1e1, log=True),
                'random_state': 42,
                'n_jobs': 1,
            }
            model = LGBMRegressor(**params)

        else:  # rf
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=100),
                'max_depth': trial.suggest_categorical('max_depth', [None, 5, 10]),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 3),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 2),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': 1,
            }
            model = RandomForestRegressor(**params)

        # 5-folds CV
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        mses = []
        for train_idx, val_idx in cv.split(X):
            m = model.__class__(**model.get_params())
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = m.predict(X.iloc[val_idx])
            mses.append(mean_squared_error(y.iloc[val_idx], pred))

        return float(np.mean(mses))

    models = {}
    hp_records = []

    for model_type in ['xgb', 'lgb', 'rf']:
        print(f"\nOptimizing {model_type.upper()}...")
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(
            lambda tr: objective(tr, model_type, X, y),
            n_trials=50,
            n_jobs=1,
            show_progress_bar=True
        )

        best_params = study.best_trial.params
        best_mse = study.best_value
        print(f"Best {model_type.upper()} params:", best_params)
        print(f"Best {model_type.upper()} MSE:", best_mse)

        if model_type == 'xgb':
            mdl = XGBRegressor(**best_params)
        elif model_type == 'lgb':
            mdl = LGBMRegressor(**best_params)
        else:
            mdl = RandomForestRegressor(**best_params)

        mdl.fit(X, y)
        models[model_type] = mdl

        dump(mdl, f"{model_type}_best_model.joblib")
        with open(f"{model_type}_best_params.json", 'w') as f:
            json.dump({
                'model_type': model_type,
                'params': best_params,
                'mse': best_mse
            }, f, indent=2)

        record = {'model_type': model_type, 'mse': best_mse}
        record.update(best_params)
        hp_records.append(record)

    df_hp = pd.DataFrame(hp_records)
    df_hp.to_csv("best_hyperparams.csv", index=False)
    print("Saved best hyperparameters (with model_type) to best_hyperparams.csv")

    # 1) Display X and y as a table
    df_display = X.copy()
    df_display['LesionRatio'] = y
    print("\n=== Sample of Features and Target ===")
    print(df_display.head().to_string(index=False))
    df_display.to_csv("features_and_target.csv", index=False)
    print("Saved full features+target table to features_and_target.csv")

    # 2) 5-Fold CV scatter plot of Predictions vs Observations
    df_hp = pd.read_csv("best_hyperparams.csv")
    best_idx = df_hp['mse'].idxmin()
    best_row = df_hp.loc[best_idx:best_idx]
    print("=== Best Hyperparameters ===")
    print(best_row.to_string(index=False))

    best_model_type = best_row.iloc[0]['model_type']
    model_file = f"{best_model_type}_best_model.joblib"
    mdl = joblib.load(model_file)

    # 3) 5-folds CV prediction
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_true, y_pred = [], []
    for train_idx, test_idx in kf.split(X):
        mdl_fold = mdl.__class__(**mdl.get_params())
        mdl_fold.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = mdl_fold.predict(X.iloc[test_idx])
        y_true.append(y.iloc[test_idx].values)
        y_pred.append(preds)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.7, s=60)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, linestyle='--', linewidth=1, zorder=0)
    plt.xlabel("Observed LesionRatio", fontsize=14, fontweight='bold')
    plt.ylabel("Predicted LesionRatio", fontsize=14, fontweight='bold')
    plt.title(
        f"{best_model_type.upper()} 5-Fold CV\n"
        f"R² = {r2:.3f},  RMSE = {rmse:.3f}",
        fontsize=16, fontweight='bold'
    )
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(f"cv_scatter_{best_model_type}.png", dpi=300)
    plt.show()

    # ———— SHAP  ————
    best_row = df_hp.loc[df_hp['mse'].idxmin()]
    best_model = best_row['model_type']
    print(f"\n>>> Best overall model is {best_model.upper()} with MSE={best_row['mse']:.6f}")

    mdl = models[best_model]

    explainer = shap.Explainer(mdl, X)
    shap_res = explainer(X)
    shap_vals = shap_res.values

    df_shap = pd.DataFrame(shap_vals, columns=X.columns)
    df_shap['PatientID'] = data['PatientID']
    out_fn = f"shap_values_best_{best_model}.csv"
    df_shap.to_csv(out_fn, index=False)
    print(f"Saved SHAP values for best model ({best_model.upper()}) to {out_fn}")

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    feat_imp = pd.Series(mean_abs_shap, index=X.columns).sort_values()

    plt.figure(figsize=(8, 6))
    feat_imp.plot(kind='barh', zorder=3)
    plt.xlabel("Mean |SHAP value|", fontsize=14, fontweight='bold')
    plt.title(f"SHAP Feature Importance ({best_model_type.upper()})", fontsize=16, fontweight='bold')
    plt.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"shap_importance_{best_model_type}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
