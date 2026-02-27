import os
import re
import time
import math
import glob
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import scipy.signal as signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import deque

SEED = 29
random.seed(SEED)
np.random.seed(SEED)



# 0) Emotion mapping
POSITIVE_IDS = {1, 2, 3, 4, 7, 15, 16, 19, 20, 21, 22, 23, 25}  # admiration..joy..etc.
NEGATIVE_IDS = {5, 6, 8, 9, 11, 12, 13, 14, 17, 18, 24, 27, 10, 26}  # anger..sadness.. etc.

def emotion_to_binary(emotion_id: int) -> int:
    """Return +1 for positive, -1 for negative."""
    if emotion_id in POSITIVE_IDS and emotion_id not in NEGATIVE_IDS:
        return +1
    if emotion_id in NEGATIVE_IDS and emotion_id not in POSITIVE_IDS:
        return -1


# 1) Config
@dataclass
class PipelineConfig:
    fs: int = 256
    # Preprocessing
    bp_low: float = 0.5
    bp_high: float = 45.0
    bp_order: int = 4
    # Windowing
    window_sec: float = 2.0
    window_hop_sec: float = 2.0
    # Features
    bands: Tuple[Tuple[str, float, float], ...] = (
        ("delta", 0.5, 4),
        ("theta", 4, 8),
        ("alpha", 8, 13),
        ("beta", 13, 30),
        ("gamma", 30, 45),
    )
    asym_pairs: Tuple[Tuple[int, int], ...] = (

    )
    # Memory controller simulation
    feature_cache_size: int = 256
    sv_cache_size: int = 256
    nvm_read_latency_ms: float = 0.2
    nvm_write_latency_ms: float = 20
    # Inference model
    kernel: str = "rbf"
    gamma: float = 0.5   # for RBF
    # Adaptive learning
    adaptive_enabled: bool = True
    margin_skip: float = 0.8
    margin_update: float = 0.2
    learning_rate: float = 0.05
    sv_budget: int = 1000
    delta_buffer_capacity: int = 32


# 2) Preprocessing
class Preprocessor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        nyq = 0.5 * cfg.fs
        self.bp_b, self.bp_a = signal.butter(cfg.bp_order,[cfg.bp_low / nyq, cfg.bp_high / nyq],btype="band")
    def run(self, x: np.ndarray) -> np.ndarray:
        y = signal.filtfilt(self.bp_b, self.bp_a, x, axis=0)
        return y



# 3) Windowing
def make_windows(x: np.ndarray, fs: int, win_sec: float, hop_sec: float) -> np.ndarray:
    W = int(round(win_sec * fs))
    H = int(round(hop_sec * fs))
    T = x.shape[0]
    if T < W:
        return np.empty((0, W, x.shape[1]), dtype=np.float32)
    n = 1 + (T - W) // H #how many windoes we need
    out = np.zeros((n, W, x.shape[1]), dtype=np.float32)
    for i in range(n):
        s = i * H
        out[i] = x[s:s+W]
    return out



# 4) Feature extraction
class FeatureExtractor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
    def _bandpower(self, psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
        idx = (freqs >= lo) & (freqs <= hi)
        if not np.any(idx):
            return np.zeros((psd.shape[1],), dtype=np.float32)
        return np.trapezoid(psd[idx, :], freqs[idx], axis=0).astype(np.float32)

    def run(self, win: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        W, C = win.shape
        freqs, psd = signal.welch(win, fs=cfg.fs, axis=0, nperseg=min(W, 256))
        feat = []
        for _, lo, hi in cfg.bands:
            bp = self._bandpower(psd, freqs, lo, hi)  # (C,)
            feat.append(float(np.mean(bp)))
            sigma2 = np.maximum(np.mean(bp), 1e-12)
            de = 0.5 * math.log(2 * math.pi * math.e * sigma2)
            feat.append(float(de))
        feat.append(float(np.mean(win)))
        feat.append(float(np.std(win)))
        feat.append(float(np.var(win)))
        if cfg.asym_pairs:
            for (li, ri) in cfg.asym_pairs:
                left = np.mean(win[:, li])
                right = np.mean(win[:, ri])
                denom = (left + right) if known_nonzero(left + right) else 1e-9
                feat.append(float((left - right) / denom))

        return np.array(feat, dtype=np.float32)

def known_nonzero(v: float) -> bool:
    return abs(v) > 1e-12

# 5) Prefetcher
class Prefetcher:
    def __init__(self, window_size: int = 10, classes: Tuple[int, ...] = (-1, +1)):
        self.window = deque(maxlen=window_size)
        self.classes = set(classes)

    def predict_next(self) -> Optional[int]:
        if not self.window:
            return None
        counts = {}
        for y in self.window:
            counts[y] = counts.get(y, 0) + 1
        max_count = max(counts.values())
        tied = {k for k, v in counts.items() if v == max_count}

        if len(tied) == 1:
            return next(iter(tied))
        for y in reversed(self.window):
            if y in tied:
                return y
        return next(iter(tied))
    def update(self, y_hat: int):
        if y_hat in self.classes:
            self.window.append(y_hat)

# 6) Caches + Memory Controller + Delta Buffer
class RingCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items: List = []

    def put(self, item):
        if len(self.items) >= self.capacity:
            self.items.pop(0)
        self.items.append(item)

    def get_all(self):
        return list(self.items)

class SVCache:
    def __init__(self, capacity: int, nvm_read_latency_ms: float):
        self.capacity = capacity
        self.lat_ms = nvm_read_latency_ms
        self.cache: Dict[int, np.ndarray] = {}
        self.lru: List[int] = []

    def _touch(self, idx: int):
        if idx in self.lru:
            self.lru.remove(idx)
        self.lru.append(idx)

    def fetch(self, idx: int, sv_from_nvm: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        if idx in self.cache:
            self._touch(idx)
            return self.cache[idx], 0.0, True

        added = self.lat_ms

        if len(self.cache) >= self.capacity:
            evict = self.lru.pop(0)
            del self.cache[evict]
        self.cache[idx] = sv_from_nvm.copy()
        self._touch(idx)
        return self.cache[idx], added, False

class DeltaBuffer:
    def __init__(self, capacity: int, nvm_write_latency_ms: float):
        self.capacity = capacity
        self.nvm_write_latency_ms = nvm_write_latency_ms
        self.sv: List[np.ndarray] = []
        self.alpha: List[float] = []

    def add(self, sv: np.ndarray, alpha: float) -> Tuple[bool, float]:
        self.sv.append(sv)
        self.alpha.append(alpha)

        if len(self.sv) >= self.capacity:
            # flush as one go
            cost = self.nvm_write_latency_ms
            return True, cost
        return False, 0.0

    def drain(self) -> Tuple[List[np.ndarray], List[float]]:
        svs, alphas = self.sv, self.alpha
        self.sv, self.alpha = [], []
        return svs, alphas

    def __len__(self):
        return len(self.sv)

# 7) Online kernel model (supports inference + light adaptive updates)
class OnlineKernelModel:
    def __init__(self, kernel: str, gamma: float, scaler: Optional[StandardScaler] = None):
        self.kernel = kernel
        self.gamma = gamma
        self.scaler = scaler
        self.nvm_sv: List[np.ndarray] = []
        self.nvm_alpha: List[float] = []
        self.b: float = 0.0
        self.delta: Optional[DeltaBuffer] = None

        self.sv_by_class = {+1: [], -1: []}

    def attach_delta_buffer(self, delta: DeltaBuffer):
        self.delta = delta

    def _k(self, x: np.ndarray, z: np.ndarray) -> float:
        if self.kernel == "linear":
            return float(np.dot(x, z))
        d2 = float(np.sum((x - z) ** 2))
        return float(math.exp(-self.gamma * d2))

    def decision(self, x: np.ndarray, sv_cache: SVCache = None) -> Tuple[float, float]:
        if self.scaler is not None:
            x = self.scaler.transform(x.reshape(1, -1)).ravel()

        fx = self.b
        extra_ms = 0.0

        if self.delta is not None and len(self.delta) > 0:
            for a, sv in zip(self.delta.alpha, self.delta.sv):
                fx += a * self._k(x, sv)
        for i, a in enumerate(self.nvm_alpha):
            if sv_cache is None:
                sv = self.nvm_sv[i]
                fx += a * self._k(x, sv)
            else:
                sv, add_ms, _hit = sv_cache.fetch(i, self.nvm_sv[i])
                extra_ms += add_ms
                fx += a * self._k(x, sv)

        return fx, extra_ms

    def predict(self, x: np.ndarray, sv_cache: SVCache = None) -> Tuple[int, float, float]:
        fx, extra_ms = self.decision(x, sv_cache)
        y_hat = +1 if fx >= 0 else -1
        return y_hat, abs(fx), extra_ms

    def initialize_from_training(self, X: np.ndarray, y: np.ndarray, max_sv: int = 500):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        self.nvm_sv = []
        self.nvm_alpha = []
        self.b = 0.0

        idxs = np.arange(len(Xs))
        np.random.shuffle(idxs)

        for idx in idxs:
            xi = Xs[idx]
            yi = int(y[idx])
            fx, _ = self.decision(xi, sv_cache=None)
            y_hat = +1 if fx >= 0 else -1
            if y_hat != yi:
                self.nvm_sv.append(xi.astype(np.float32))
                self.nvm_alpha.append(float(yi))
                if len(self.nvm_sv) >= max_sv:
                    break

        self._rebuild_class_index()

    def _rebuild_class_index(self):
        self.sv_by_class = {+1: [], -1: []}
        for i, a in enumerate(self.nvm_alpha):
            cls = +1 if a >= 0 else -1
            self.sv_by_class[cls].append(i)

    def adaptive_update(self, x: np.ndarray, y_hat: int, margin: float, cfg: PipelineConfig) -> Tuple[str, float]:
        if self.scaler is not None:
            x = self.scaler.transform(x.reshape(1, -1)).ravel()
        if margin > cfg.margin_skip:
            return "skip", 0.0
        scale = 1.0 if margin < cfg.margin_update else 0.5
        return self._add_sv_delta(x, y_hat, cfg, scale=scale)
    def _add_sv_delta(self, x: np.ndarray, y_hat: int, cfg: PipelineConfig, scale: float) -> Tuple[str, float]:
        if self.delta is None:
            raise RuntimeError("Delta buffer not attached. Call attach_delta_buffer().")
        total = len(self.nvm_sv) + len(self.delta)
        if total >= cfg.sv_budget:
            if self.nvm_sv:
                self.nvm_sv.pop(0)
                self.nvm_alpha.pop(0)
            else:
                self.delta.sv.pop(0)
                self.delta.alpha.pop(0)
        new_sv = x.astype(np.float32)
        new_alpha = float(cfg.learning_rate * scale * y_hat)
        flushed, write_ms = self.delta.add(new_sv, new_alpha)
        if flushed:
            svs, alphas = self.delta.drain()
            self.nvm_sv.extend(svs)
            self.nvm_alpha.extend(alphas)
            self._rebuild_class_index()
            return "flush", write_ms

        return "buffer", 0.0


# 8) Dataset loader (txt files)
FILE_RE = re.compile(r"(?P<p>\d+)[_\-](?P<e>\d+)\.0\.txt$")

def load_txt_eeg(file_path: str) -> np.ndarray:
    """
    Returns x shape (T, C=14) float32
    """
    data = np.loadtxt(file_path, dtype=np.float32)
    # If file has (T, 14) it's correct; if transposed, fix.
    if data.ndim == 1:
        raise ValueError(f"{file_path} appears to have 1 column; expected 14 columns.")
    if data.shape[1] != 14 and data.shape[0] == 14:
        data = data.T
    if data.shape[1] != 14:
        raise ValueError(f"{file_path} has shape {data.shape}; expected 14 columns.")
    return data

def list_dataset_files(root_dir: str) -> List[Tuple[str, int, int]]:
    out = []
    for fp in glob.glob(os.path.join(root_dir, "*.txt")):
        m = FILE_RE.search(os.path.basename(fp))
        if not m:
            continue
        p = int(m.group("p"))
        e = int(m.group("e"))
        out.append((fp, p, e))
    if not out:
        raise FileNotFoundError(
            f"No files matched pattern like '12_5.0.txt' in {root_dir}. "
            f"Check your filenames/path."
        )
    return out

# 9) Pipeline Simulator
class BCIPipelineSimulator:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.pre = Preprocessor(cfg)
        self.fe = FeatureExtractor(cfg)
        self.feature_cache = RingCache(cfg.feature_cache_size)
        self.sv_cache = SVCache(cfg.sv_cache_size, cfg.nvm_read_latency_ms)
        self.prefetcher = Prefetcher()
        self.model = OnlineKernelModel(kernel=cfg.kernel, gamma=cfg.gamma, scaler=None)
        self.delta = DeltaBuffer(cfg.delta_buffer_capacity, cfg.nvm_write_latency_ms)
        self.model.attach_delta_buffer(self.delta)

    def offline_initialize(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.initialize_from_training(X_train, y_train, max_sv=min(500, self.cfg.sv_budget))

    def _prefetch_sv_for_class(self, cls: int) -> float:
        if cls not in self.model.sv_by_class:
            return 0.0

        extra = 0.0
        candidates = self.model.sv_by_class[cls][: min(32, len(self.model.sv_by_class[cls]))]
        for idx in candidates:
            _sv, add_ms, _hit = self.sv_cache.fetch(idx, self.model.nvm_sv[idx])
            extra += add_ms
        return extra

    def process_file_streaming(self, file_path: str, true_label: Optional[int]) -> Dict[str, float]:
        cfg = self.cfg
        raw = load_txt_eeg(file_path)
        t0 = time.perf_counter()

        # Preprocessing
        tp0 = time.perf_counter()
        x = self.pre.run(raw)
        tp1 = time.perf_counter()

        tw0 = time.perf_counter()
        wins = make_windows(x, cfg.fs, cfg.window_sec, cfg.window_hop_sec)
        tw1 = time.perf_counter()

        # Streaming over windows
        correct = 0
        total = 0

        # latency accumulators
        feat_ms = 0.0
        infer_ms = 0.0
        mem_ms = 0.0
        learn_ms = 0.0

        for w in wins:
            # Feature extraction
            tf0 = time.perf_counter()
            fv = self.fe.run(w)
            tf1 = time.perf_counter()
            feat_ms += (tf1 - tf0) * 1e3

            # Feature cache write
            self.feature_cache.put(fv)

            # Prefetch
            pred_next = self.prefetcher.predict_next()
            if pred_next is not None:
                tm0 = time.perf_counter()
                mem_ms += self._prefetch_sv_for_class(pred_next)
                tm1 = time.perf_counter()
                mem_ms += (tm1 - tm0) * 1e3 * 0.0  # compute is negligible; keep 0

            # Inference
            ti0 = time.perf_counter()
            y_hat, margin, extra_mem = self.model.predict(fv, sv_cache=self.sv_cache)
            ti1 = time.perf_counter()
            infer_ms += (ti1 - ti0) * 1e3
            mem_ms += extra_mem

            self.prefetcher.update(y_hat)
            if true_label is not None:
                total += 1
                if y_hat == true_label:
                    correct += 1
            # Adaptive learning
            if cfg.adaptive_enabled:
                tl0 = time.perf_counter()
                mode, wms = self.model.adaptive_update(fv, y_hat, margin, cfg)
                tl1 = time.perf_counter()
                learn_ms += (tl1 - tl0) * 1e3
                mem_ms += wms
        t1 = time.perf_counter()

        out = {
            "windows": float(len(wins)),
            "preprocess_ms_total": (tp1 - tp0) * 1e3,
            "windowing_ms_total": (tw1 - tw0) * 1e3,
            "feature_ms_total": feat_ms,
            "inference_ms_total": infer_ms,
            "memory_ms_total": mem_ms,
            "learning_ms_total": learn_ms,
            "end_to_end_ms_total": (t1 - t0) * 1e3,
        }
        if true_label is not None and total > 0:
            out["accuracy"] = correct / total
        else:
            out["accuracy"] = float("nan")
        return out

# 10) Build train set from files (offline init)
def build_training_set(files: List[Tuple[str, int, int]], cfg: PipelineConfig,
                       max_files: int = 50, max_windows_per_file: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    pre = Preprocessor(cfg)
    fe = FeatureExtractor(cfg)

    random.shuffle(files)
    files = files[:max_files]

    X_list = []
    y_list = []

    for fp, _p, emotion_id in files:
        y = emotion_to_binary(emotion_id)

        raw = load_txt_eeg(fp)
        x = pre.run(raw)
        wins = make_windows(x, cfg.fs, cfg.window_sec, cfg.window_hop_sec)
        if len(wins) == 0:
            continue
        take = min(max_windows_per_file, len(wins))
        idxs = np.linspace(0, len(wins) - 1, num=take, dtype=int)

        for i in idxs:
            fv = fe.run(wins[i])
            X_list.append(fv)
            y_list.append(y)
    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


# 11) Main
def main(dataset_dir: str):
    cfg = PipelineConfig(
        fs=256,
        bp_low=0.5,
        bp_high=45.0,
        window_sec=5,
        window_hop_sec=5,
        kernel="rbf",
        gamma=0.5,
        adaptive_enabled=True
    )

    files = list_dataset_files(dataset_dir)


    X, y = build_training_set(files, cfg, max_files=80, max_windows_per_file=25)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    sim = BCIPipelineSimulator(cfg)
    sim.offline_initialize(X_train, y_train)

    test_files = files[:10]
    all_stats = []

    for fp, p, emotion_id in test_files:
        y_true = emotion_to_binary(emotion_id)

        stats = sim.process_file_streaming(fp, true_label=y_true)
        all_stats.append(stats)

        print(f"\n[ONLINE] File={os.path.basename(fp)} (P{p}, emotion {emotion_id}, y={y_true})")
        print(f"  windows: {int(stats['windows'])}")
        print(f"  accuracy: {stats['accuracy']:.3f}")
        print(f"  preprocess_ms_total: {stats['preprocess_ms_total']:.2f}")
        print(f"  feature_ms_total: {stats['feature_ms_total']:.2f}")
        print(f"  inference_ms_total: {stats['inference_ms_total']:.2f}")
        print(f"  memory_ms_total (incl. SV cache misses): {stats['memory_ms_total']:.2f}")
        print(f"  learning_ms_total: {stats['learning_ms_total']:.2f}")
        print(f"  end_to_end_ms_total: {stats['end_to_end_ms_total']:.2f}")

    def avg(key: str) -> float:
        vals = [s[key] for s in all_stats if not math.isnan(s[key])]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n[SUMMARY OVER TEST FILES]")
    print(f"  avg accuracy: {avg('accuracy'):.3f}")
    print(f"  avg feature_ms_total: {avg('feature_ms_total'):.2f}")
    print(f"  avg inference_ms_total: {avg('inference_ms_total'):.2f}")
    print(f"  avg memory_ms_total: {avg('memory_ms_total'):.2f}")
    print(f"  avg learning_ms_total: {avg('learning_ms_total'):.2f}")
    print(f"  avg end_to_end_ms_total: {avg('end_to_end_ms_total'):.2f}")
    print(f"  final SV count (after adaptive updates): {len(sim.model.nvm_sv) + len(sim.model.delta)}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
