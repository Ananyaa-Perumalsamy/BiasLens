"""
core/dataset.py
───────────────
Universal dataset loader.
Accepts ANY CSV — auto-detects sensitive attributes,
outcome columns, and feature columns.
Also supports image datasets with CSV or filename-based metadata.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import re
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# ── Keywords used to auto-detect sensitive / demographic columns ──
SENSITIVE_KEYWORDS = [
    "gender", "sex", "race", "ethnicity", "age", "religion",
    "nationality", "disability", "marital", "color", "colour",
    "caste", "tribe", "origin", "orientation",
]

# Keywords to auto-detect outcome / decision columns
OUTCOME_KEYWORDS = [
    "hired", "approved", "rejected", "outcome", "decision",
    "label", "target", "result", "status", "score", "recid",
    "convicted", "granted", "denied", "pass", "fail", "default",
    "fraud", "risk", "class", "y",
]


# ══════════════════════════════════════════════════════════════
#  Column scanner
# ══════════════════════════════════════════════════════════════

def scan_columns(df: pd.DataFrame) -> Dict:
    """
    Scans a DataFrame and returns suggested roles for each column:
      - sensitive  : likely demographic / protected attribute
      - outcome    : likely the decision / prediction column
      - feature    : everything else
    Also returns cardinality and sample values for each column.
    """
    suggestions = {}

    for col in df.columns:
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        nunique = df[col].nunique()
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().unique()[:6].tolist()

        # Guess role
        role = "feature"
        confidence = "low"

        # Check sensitive keywords
        for kw in SENSITIVE_KEYWORDS:
            if kw in col_lower:
                role = "sensitive"
                confidence = "high"
                break

        # Check outcome keywords
        if role == "feature":
            for kw in OUTCOME_KEYWORDS:
                if kw in col_lower:
                    role = "outcome"
                    confidence = "high"
                    break

        # Heuristic: low-cardinality categorical with no keyword match
        # could still be sensitive
        if role == "feature" and nunique <= 7 and dtype == "object":
            role = "sensitive"
            confidence = "medium"

        # Binary column with no keyword = possible outcome
        if role == "feature" and nunique == 2:
            role = "outcome"
            confidence = "medium"

        suggestions[col] = {
            "role":        role,
            "confidence":  confidence,
            "dtype":       dtype,
            "nunique":     nunique,
            "sample_vals": sample_vals,
        }

    return suggestions


# ══════════════════════════════════════════════════════════════
#  Group index builder
# ══════════════════════════════════════════════════════════════

def build_group_indices(df: pd.DataFrame,
                        sensitive_col: str) -> Dict[str, List[int]]:
    """
    Returns {group_name: [row_indices]} for a given sensitive column.
    """
    groups = {}
    for grp, sub in df.groupby(sensitive_col):
        groups[str(grp)] = sub.index.tolist()
    return groups


# ══════════════════════════════════════════════════════════════
#  Dataset summary
# ══════════════════════════════════════════════════════════════

def dataset_summary(df: pd.DataFrame,
                    sensitive_cols: List[str],
                    outcome_col: str) -> Dict:
    """
    Returns a quick summary dict:
      - shape
      - missing values
      - class distribution of outcome
      - group distributions for each sensitive col
    """
    summary = {
        "rows":    len(df),
        "columns": len(df.columns),
        "missing": int(df.isnull().sum().sum()),
        "outcome_distribution": {},
        "group_distributions":  {},
    }

    if outcome_col and outcome_col in df.columns:
        summary["outcome_distribution"] = (
            df[outcome_col].value_counts().to_dict()
        )

    for col in sensitive_cols:
        if col in df.columns:
            summary["group_distributions"][col] = (
                df[col].value_counts().to_dict()
            )

    return summary


# ══════════════════════════════════════════════════════════════
#  Label encoder (for non-numeric outcome columns)
# ══════════════════════════════════════════════════════════════

def encode_outcome(df: pd.DataFrame,
                   outcome_col: str) -> Tuple[np.ndarray, Dict]:
    """
    Encodes the outcome column to integers.
    Returns (encoded_array, label_map).
    """
    series = df[outcome_col].astype(str)
    unique_vals = sorted(series.dropna().unique().tolist())
    label_map = {v: i for i, v in enumerate(unique_vals)}
    encoded = series.map(label_map).fillna(-1).astype(int).values
    return encoded, label_map


# ══════════════════════════════════════════════════════════════
#  Known demo datasets
# ══════════════════════════════════════════════════════════════

DEMO_DATASETS = {
    "Adult Income (UCI)": {
        "url": "https://raw.githubusercontent.com/dssg/fairness_tutorial/master/data/processed/adult.csv",
        "description": "Census income data — predict if income >$50K. Known gender & race bias.",
        "sensitive_cols": ["sex", "race"],
        "outcome_col": "income",
    },
    "COMPAS Recidivism": {
        "url": "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
        "description": "Criminal risk scores — famous ProPublica race bias study.",
        "sensitive_cols": ["race", "sex"],
        "outcome_col": "two_year_recid",
    },
    "German Credit": {
        "url": "https://raw.githubusercontent.com/dssg/fairness_tutorial/master/data/processed/german_credit.csv",
        "description": "Loan approval dataset — age & gender bias in credit decisions.",
        "sensitive_cols": ["sex", "age"],
        "outcome_col": "credit_risk",
    },
}


# ══════════════════════════════════════════════════════════════
#  Image Dataset Classes
# ══════════════════════════════════════════════════════════════

class ImageDataset(Dataset):
    """PyTorch Dataset for lazy image loading with demographic metadata."""

    def __init__(self, image_paths: List[str],
                 labels: np.ndarray,
                 sensitive_attrs: Dict[str, np.ndarray],
                 transform=None):
        """
        Args:
            image_paths: List of image file paths
            labels: Outcome labels (N,)
            sensitive_attrs: Dict of {attr_name: values} for fairness analysis
            transform: torchvision transforms to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.sensitive_attrs = sensitive_attrs
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")

        label = self.labels[idx]
        attrs = {k: v[idx] for k, v in self.sensitive_attrs.items()}

        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long),
            'index': idx,
            **{f'attr_{k}': torch.tensor(v, dtype=torch.long)
               for k, v in attrs.items()}
        }


def parse_utk_filename(filename: str) -> Dict:
    """
    Parse UTK face dataset filename format: age_gender_race_datetime.jpg
    Returns: {age, gender, race} or None if parsing fails.

    age: 0-116
    gender: 0=M, 1=F
    race: 0=White, 1=Black, 2=Asian, 3=Indian, 4=Other
    """
    pattern = r'(\d+)_([01])_([0-4])_\d+\.jpg'
    match = re.match(pattern, filename)
    if match:
        return {
            'age': int(match.group(1)),
            'gender': int(match.group(2)),
            'race': int(match.group(3))
        }
    return None


def load_image_metadata_from_csv(csv_path: str,
                                image_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load image metadata from CSV file.
    CSV must have 'file' column with relative image paths.
    Returns: (dataframe with metadata, list of full image paths)
    """
    df = pd.read_csv(csv_path)

    if 'file' not in df.columns:
        raise ValueError("CSV must have 'file' column with image paths")

    image_paths = []
    valid_indices = []

    for idx, filename in enumerate(df['file']):
        full_path = os.path.join(image_dir, filename)
        if os.path.exists(full_path):
            image_paths.append(full_path)
            valid_indices.append(idx)
        else:
            print(f"Warning: Image not found: {full_path}")

    df = df.iloc[valid_indices].reset_index(drop=True)
    return df, image_paths


def load_image_metadata_from_filenames(image_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parse image metadata from filenames (UTK format).
    Returns: (dataframe with parsed metadata, list of full image paths)
    """
    records = []
    image_paths = []

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(('.jpg', '.png', '.bmp')):
            continue

        parsed = parse_utk_filename(filename)
        if parsed is None:
            print(f"Warning: Could not parse filename: {filename}")
            continue

        full_path = os.path.join(image_dir, filename)
        records.append({
            'file': filename,
            'age': parsed['age'],
            'gender': parsed['gender'],
            'race': parsed['race']
        })
        image_paths.append(full_path)

    if not records:
        raise ValueError(f"No valid images found in {image_dir}")

    df = pd.DataFrame(records)
    return df, image_paths


def build_dataloaders(image_dir: str,
                      metadata_source: str = None,
                      metadata_path: Optional[str] = None,
                      metadata_df: Optional[pd.DataFrame] = None,
                      image_paths: Optional[List[str]] = None,
                      sensitive_cols: Optional[List[str]] = None,
                      outcome_col: Optional[str] = None,
                      batch_size: int = 32,
                      val_split: float = 0.2,
                      stratify_by: Optional[List[str]] = None,
                      device: str = 'cpu') -> Tuple:
    """
    Build PyTorch DataLoaders for image datasets with fairness support.

    Args:
        image_dir: Directory containing images
        metadata_source: 'csv' or 'filenames' (ignored if metadata_df provided)
        metadata_path: Path to CSV if metadata_source=='csv'
        metadata_df: Pre-loaded dataframe (used if provided, else load from metadata_source)
        image_paths: Pre-computed image paths (used if provided)
        sensitive_cols: List of sensitive attribute columns (auto-detected if None)
        outcome_col: Name of outcome column (auto-detected if None)
        batch_size: DataLoader batch size
        val_split: Validation split ratio
        stratify_by: List of columns to stratify by (for balanced splits)
        device: 'cpu' or 'cuda'

    Returns:
        (train_loader, val_loader, num_classes, label_map, metadata_df, group_indices)
    """
    # Load metadata if not provided
    if metadata_df is None:
        if metadata_source == 'csv':
            if metadata_path is None:
                raise ValueError("metadata_path required for metadata_source='csv'")
            df, image_paths = load_image_metadata_from_csv(metadata_path, image_dir)
        elif metadata_source == 'filenames':
            df, image_paths = load_image_metadata_from_filenames(image_dir)
        else:
            raise ValueError(f"Unknown metadata_source: {metadata_source}")
    else:
        df = metadata_df
        # Generate image_paths from df if not provided
        if image_paths is None:
            image_paths = [os.path.join(image_dir, f) for f in df.get('file', [])]

    if len(df) == 0:
        raise ValueError("No valid images found")

    # Auto-detect sensitive and outcome columns
    if sensitive_cols is None or outcome_col is None:
        suggestions = scan_columns(df)
        if sensitive_cols is None:
            sensitive_cols = [col for col, info in suggestions.items()
                            if info['role'] == 'sensitive' and col != 'file']
        if outcome_col is None:
            for col, info in suggestions.items():
                if info['role'] == 'outcome':
                    outcome_col = col
                    break

    if not sensitive_cols:
        sensitive_cols = []
    if outcome_col is None:
        raise ValueError("No outcome column detected or provided")

    # Encode outcome
    y_encoded, label_map = encode_outcome(df, outcome_col)
    num_classes = len(label_map)

    # Build group indices for fairness
    group_indices = {}
    for col in sensitive_cols:
        if col in df.columns:
            group_indices[col] = build_group_indices(df, col)

    # Stratified train/val split
    if stratify_by is None:
        stratify_by = sensitive_cols if sensitive_cols else None

    train_idx, val_idx = None, None

    if stratify_by and all(col in df.columns for col in stratify_by):
        # Try stratifying by primary demographic attribute first
        for col in stratify_by:
            try:
                stratify_col = df[col]
                train_idx, val_idx = train_test_split(
                    range(len(df)),
                    test_size=val_split,
                    stratify=stratify_col,
                    random_state=42
                )
                break
            except ValueError:
                continue

    # Fall back to random split if stratification fails
    if train_idx is None:
        train_idx, val_idx = train_test_split(
            range(len(df)),
            test_size=val_split,
            random_state=42
        )

    train_paths = [image_paths[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]

    train_labels = y_encoded[train_idx]
    val_labels = y_encoded[val_idx]

    train_sensitive = {col: df[col].values[train_idx]
                     for col in sensitive_cols}
    val_sensitive = {col: df[col].values[val_idx]
                    for col in sensitive_cols}

    # Create PyTorch datasets
    train_dataset = ImageDataset(train_paths, train_labels, train_sensitive)
    val_dataset = ImageDataset(val_paths, val_labels, val_sensitive)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, num_classes, label_map, df, group_indices
