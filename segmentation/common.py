"""
Shared helpers for the road segmentation stage.

Used by prepare_split.py, train_unet.py and evaluate.py.
"""
import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet statistics: the ResNet-34 encoder was pre-trained with these.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- utilities
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_train_dir(data_dir: str) -> str:
    """Find the DeepGlobe folder that holds *_sat.jpg + *_mask.png pairs.

    Works whether the zip was extracted as data_dir/train or data_dir/archive/train.
    """
    candidates = []
    for root, _dirs, files in os.walk(data_dir):
        has_mask = any(f.endswith("_mask.png") for f in files)
        has_sat = any(f.endswith("_sat.jpg") for f in files)
        if has_mask and has_sat:
            candidates.append(root)
    if not candidates:
        raise FileNotFoundError(
            f"No folder with *_sat.jpg and *_mask.png found under {data_dir}. "
            "Check that the DeepGlobe zip is fully extracted there."
        )
    # Prefer a folder literally called 'train'
    candidates.sort(key=lambda p: (os.path.basename(p).lower() != "train", len(p)))
    return candidates[0]


def list_pair_ids(train_dir: str):
    """Return (ids_with_both_files, sat_without_mask, mask_without_sat)."""
    sats = {os.path.basename(p)[: -len("_sat.jpg")] for p in glob.glob(os.path.join(train_dir, "*_sat.jpg"))}
    masks = {os.path.basename(p)[: -len("_mask.png")] for p in glob.glob(os.path.join(train_dir, "*_mask.png"))}
    return sorted(sats & masks), sorted(sats - masks), sorted(masks - sats)


def read_ids(path: str):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def read_image(path: str) -> np.ndarray:
    """Read an image as RGB uint8 (H, W, 3)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: str) -> np.ndarray:
    """Read a DeepGlobe mask as uint8 0/1 (H, W). Roads are white (255)."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.uint8)


def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    """RGB uint8 (H, W, 3) -> normalised float tensor (3, H, W)."""
    x = (img_rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


# ------------------------------------------------------------------ dataset
class RoadDataset(Dataset):
    """DeepGlobe image/mask pairs.

    crop=None  -> return the full image (used for validation/testing)
    crop=512   -> return a random 512x512 crop (used for training)
    """

    def __init__(self, image_dir, ids, crop=None, augment=False, min_road=0.01, tries=3):
        self.image_dir = image_dir
        self.ids = list(ids)
        self.crop = crop
        self.augment = augment
        self.min_road = min_road
        self.tries = tries

    def __len__(self):
        return len(self.ids)

    def _random_crop(self, img, mask):
        h, w = mask.shape
        c = self.crop
        best = None
        # Try a few crops and keep the first one with enough road pixels,
        # so the model does not see mostly empty background.
        for _ in range(self.tries):
            y = random.randint(0, h - c)
            x = random.randint(0, w - c)
            m = mask[y:y + c, x:x + c]
            best = (y, x)
            if m.mean() >= self.min_road:
                break
        y, x = best
        return img[y:y + c, x:x + c], mask[y:y + c, x:x + c]

    def __getitem__(self, i):
        id_ = self.ids[i]
        img = read_image(os.path.join(self.image_dir, f"{id_}_sat.jpg"))
        mask = read_mask(os.path.join(self.image_dir, f"{id_}_mask.png"))

        if self.crop is not None:
            img, mask = self._random_crop(img, mask)

        if self.augment:
            # Roads have no "up" direction in satellite images, so flips and
            # 90-degree rotations create valid new training examples.
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                img, mask = img[::-1], mask[::-1]
            k = random.randint(0, 3)
            img, mask = np.rot90(img, k), np.rot90(mask, k)
            # Mild brightness/contrast jitter
            if random.random() < 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-15, 15)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        x = to_tensor(img)
        y = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        return x, y


# -------------------------------------------------------------------- model
def build_model(pretrained: bool = True):
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,  # one output channel = road probability (after sigmoid)
    )


def load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


# ------------------------------------------------------------------ metrics
@torch.no_grad()
def confusion_counts(probs: torch.Tensor, target: torch.Tensor, thr: float):
    pred = probs >= thr
    tgt = target >= 0.5
    tp = (pred & tgt).sum().item()
    fp = (pred & ~tgt).sum().item()
    fn = (~pred & tgt).sum().item()
    return tp, fp, fn


def metrics_from_counts(tp, fp, fn):
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------- inference
@torch.no_grad()
def predict_probs(model, img_rgb: np.ndarray, device, tile: int = 1024, overlap: int = 128) -> np.ndarray:
    """Road probability map (H, W) float32 for an image of ANY size.

    Small images: padded to a multiple of 32 and predicted in one pass.
    Large images: predicted in overlapping tiles that are averaged.
    """
    use_amp = device.type == "cuda"
    H, W = img_rgb.shape[:2]

    def run(batch_img):
        x = to_tensor(batch_img).unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
        return torch.sigmoid(logits.float())[0, 0].cpu().numpy()

    if H <= tile and W <= tile:
        ph = (32 - H % 32) % 32
        pw = (32 - W % 32) % 32
        padded = np.pad(img_rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        return run(padded)[:H, :W]

    # Sliding window
    ph = max(0, tile - H)
    pw = max(0, tile - W)
    padded = np.pad(img_rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    Hp, Wp = padded.shape[:2]
    step = tile - overlap
    ys = list(range(0, Hp - tile + 1, step))
    xs = list(range(0, Wp - tile + 1, step))
    if ys[-1] != Hp - tile:
        ys.append(Hp - tile)
    if xs[-1] != Wp - tile:
        xs.append(Wp - tile)

    acc = np.zeros((Hp, Wp), dtype=np.float32)
    cnt = np.zeros((Hp, Wp), dtype=np.float32)
    for y in ys:
        for x in xs:
            acc[y:y + tile, x:x + tile] += run(padded[y:y + tile, x:x + tile])
            cnt[y:y + tile, x:x + tile] += 1
    return (acc / cnt)[:H, :W]
