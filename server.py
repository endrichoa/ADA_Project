
# server.py

import os
import numpy as np
import librosa
import tempfile
from flask import Flask, request, jsonify
from dataclasses import dataclass
from typing import Literal, Dict, List, Tuple, Optional
from copy import deepcopy

# ----------------------------
# Config / Parameters
# ----------------------------

@dataclass
class KeyDetectParams:
    sr: int = 22050
    hop_length: int = 4096
    chroma_type: Literal['cqt', 'stft', 'cens'] = 'cqt'
    use_hpss: bool = True
    smoothing: int = 9
    ref_tuning: float = 0.0
    pitch_shift: float = 0.0
    beat_engine: Literal['librosa', 'aubio'] = 'librosa'
    use_pychord: bool = False

# Krumhansl-Schmuckler key profiles
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66,
2.29, 2.88], dtype=float)
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69,
3.34, 3.17], dtype=float)
PITCH_CLASS_NAMES = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#',
'A', 'A#', 'B'])

def rotate_profile(profile: np.ndarray, n: int) -> np.ndarray:
    return np.roll(profile, n)

def key_templates() -> Dict[str, np.ndarray]:
    templates = {}
    for i, name in enumerate(PITCH_CLASS_NAMES):
        templates[f"{name} major"] = rotate_profile(KS_MAJOR, i)
        templates[f"{name} minor"] = rotate_profile(KS_MINOR, i)
    return templates

# ----------------------------
# Optional aubio helpers
# ----------------------------

def _beats_from_aubio(path: str, sr: int, hop_length: int) -> Optional[np.ndarray]:
    try:
        import aubio  # type: ignore
    except Exception:
        return None
    try:
        src = aubio.source(path, samplerate=sr, hop_size=hop_length)
        o = aubio.tempo(method='default', buf_size=hop_length * 2, hop_size=hop_length, samplerate=sr)
        beat_times: List[float] = []
        while True:
            samples, read = src()
            if o(samples):
                t = o.get_last_s()
                beat_times.append(float(t))
            if read < src.hop_size:
                break
        if not beat_times:
            return None
        beat_frames = librosa.time_to_frames(np.asarray(beat_times,
dtype=float), sr=sr, hop_length=hop_length)
        return beat_frames.astype(int)
    except Exception:
        return None

def _bpm_from_aubio(path: str, sr: int, hop_length: int) -> Optional[Tuple[float, List[float]]]:
    try:
        import aubio  # type: ignore
    except Exception:
        return None
    try:
        src = aubio.source(path, samplerate=sr, hop_size=hop_length)
        o = aubio.tempo(method='default', buf_size=hop_length * 2, hop_size=hop_length, samplerate=sr)
        beat_times: List[float] = []
        while True:
            samples, read = src()
            if o(samples):
                t = o.get_last_s()
                beat_times.append(float(t))
            if read < src.hop_size:
                break
        bpm = float(o.get_bpm()) if hasattr(o, 'get_bpm') else 0.0
        if bpm <= 0.0 and len(beat_times) >= 2:
            intervals = np.diff(np.asarray(beat_times))
            med = float(np.median(intervals)) if intervals.size else 0.0
            if med > 0:
                bpm = 60.0 / med
        if not beat_times and bpm <= 0:
            return None
        return bpm, beat_times
    except Exception:
        return None

# ----------------------------
# Audio + Chroma
# ----------------------------

def load_audio(path: str, p: KeyDetectParams):
    y, sr = librosa.load(path, sr=p.sr, mono=True)
    if abs(p.pitch_shift) > 1e-6:
        y = librosa.effects.pitch_shift(y, sr, p.pitch_shift)
    return y, sr

def drop_low_energy_frames(C, thresh_quantile=0.2):
    e = np.sum(C, axis=0)
    thr = np.quantile(e, thresh_quantile)
    keep = e >= max(thr, 1e-9)
    return C[:, keep], keep

def compute_chroma_beats(y, sr, p: KeyDetectParams, source_path: Optional[str] = None):
    if p.use_hpss:
        y = librosa.effects.harmonic(y)
    est_tuning = librosa.estimate_tuning(y=y, sr=sr, n_fft=8192)
    auto_ref_semitones = float(np.asarray(est_tuning).squeeze())  # ensure scalar

    if p.chroma_type == 'cqt':
        C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=p.hop_length)
    elif p.chroma_type == 'stft':
        S = np.abs(librosa.stft(y, n_fft=8192, hop_length=p.hop_length)) ** 2
        C = librosa.feature.chroma_stft(S=S, sr=sr, hop_length=p.hop_length)
    else:
        C = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=p.hop_length)

    beats: Optional[np.ndarray] = None
    tempo_bpm: float = 0.0
    if p.beat_engine == 'aubio' and source_path is not None:
        beats = _beats_from_aubio(source_path, sr, p.hop_length)
    if beats is None:
        _tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=p.hop_length, units='frames')
        tempo_bpm = float(np.asarray(_tempo).squeeze()) if _tempo is not None else 0.0
    else:
        bt = librosa.frames_to_time(beats, sr=sr, hop_length=p.hop_length)
        if len(bt) >= 2:
            intervals = np.diff(bt)
            med = float(np.median(intervals)) if intervals.size else 0.0
            if med > 0:
                tempo_bpm = 60.0 / med

    if beats is None:
        beats = np.array([], dtype=int)

    if len(beats) >= 4:
        Cb = librosa.util.sync(C, beats, aggregate=np.median)
        beat_frames = beats
    else:
        Cb = C
        beat_frames = np.arange(C.shape[1])

    # Normalize chroma per frame with epsilon
    Cb = Cb / (np.sum(Cb, axis=0, keepdims=True) + 1e-9)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=p.hop_length)
    return Cb, beat_frames, beat_times, auto_ref_semitones, tempo_bpm

# ----------------------------
# BPM utilities (robust estimation)
# ----------------------------

def _ensure_bpm_range(bpm: float, lo: float = 80.0, hi: float = 190.0) -> float:
    if bpm <= 0 or not np.isfinite(bpm):
        return 0.0
    for _ in range(5):
        if bpm < lo:
            bpm *= 2.0
        elif bpm > hi:
            bpm /= 2.0
        else:
            break
    return float(bpm)

def _bpm_from_intervals(beat_times: np.ndarray) -> float:
    if beat_times is None or len(beat_times) < 2:
        return 0.0
    intervals = np.diff(np.asarray(beat_times, dtype=float))
    intervals = intervals[intervals > 1e-3]
    if intervals.size == 0:
        return 0.0
    med = float(np.median(intervals))
    if med <= 0:
        return 0.0
    bpm = 60.0 / med
    return _ensure_bpm_range(bpm)

def robust_bpm(y: np.ndarray, sr: int, hop_length: int, beat_times: Optional[np.ndarray] = None) -> Tuple[int, float]:
    try:
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        # Instantaneous tempo series in BPM
        tseries = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length, aggregate=None)
        tseries = np.asarray(tseries, dtype=float)
        tseries = tseries[np.isfinite(tseries)]
        tseries = tseries[(tseries >= 40.0) & (tseries <= 220.0)]
        mode_est = 0.0
        if tseries.size > 0:
            # Histogram mode (1 BPM bins)
            bins = np.arange(40.0, 221.0, 1.0)
            hist, edges = np.histogram(tseries, bins=bins)
            if hist.sum() > 0:
                idx = int(np.argmax(hist))
                mode_est = float((edges[idx] + edges[idx+1]) * 0.5)
        beat_est = _bpm_from_intervals(beat_times) if beat_times is not None else 0.0
        # Prefer beat-derived BPM, nudged toward tempo mode estimate
        cand = beat_est if beat_est > 0 else mode_est
        if cand > 0 and mode_est > 0:
            # Blend if within 20%
            if abs(mode_est - cand) / max(cand, 1e-6) <= 0.2:
                cand = 0.5 * (cand + mode_est)
        cand = _ensure_bpm_range(cand)

        # Confidence based on beat-interval consistency
        conf = 0.0
        if beat_times is not None and len(beat_times) >= 3:
            iv = np.diff(np.asarray(beat_times))
            tempos = 60.0 / np.maximum(iv, 1e-6)
            tmed = float(np.median(tempos))
            tv = float(np.std(tempos) / (tmed + 1e-9))
            conf = float(max(0.0, 1.0 - min(tv, 1.0)))
        return int(round(cand)) if cand > 0 else 0, float(conf)
    except Exception:
        return 0, 0.0

# ----------------------------
# Camelot Wheel helpers
# ----------------------------

_CAMELOT_MAJOR = {
    'C': '8B', 'G': '9B', 'D': '10B', 'A': '11B', 'E': '12B', 'B': '1B',
    'F#': '2B', 'C#': '3B', 'G#': '4B', 'D#': '5B', 'A#': '6B', 'F': '7B'
}
_CAMELOT_MINOR = {
    'A': '8A', 'E': '9A', 'B': '10A', 'F#': '11A', 'C#': '12A', 'G#': '1A',
    'D#': '2A', 'A#': '3A', 'F': '4A', 'C': '5A', 'G': '6A', 'D': '7A'
}

_CAMELOT_TO_KEY = {**{v: f"{k} major" for k, v in _CAMELOT_MAJOR.items()}, **{v: f"{k} minor" for k, v in _CAMELOT_MINOR.items()}}

def key_to_camelot(key_str: str) -> Optional[str]:
    try:
        parts = key_str.strip().split()
        if len(parts) < 2:
            return None
        root, mode = parts[0], parts[1].lower()
        if 'maj' in mode:
            return _CAMELOT_MAJOR.get(root)
        else:
            return _CAMELOT_MINOR.get(root)
    except Exception:
        return None

def camelot_neighbors(code: str) -> List[str]:
    try:
        num = int(code[:-1])
        mode = code[-1].upper()  # 'A' or 'B'
        same_num_other_mode = f"{num}{'B' if mode == 'A' else 'A'}"
        minus = 12 if num == 1 else num - 1
        plus = 1 if num == 12 else num + 1
        return [f"{minus}{mode}", same_num_other_mode, f"{plus}{mode}"]
    except Exception:
        return []

# ----------------------------
# Key + Chord detection
# ----------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))

def estimate_key(C: np.ndarray, p: KeyDetectParams) -> dict:
    pcp = np.mean(C, axis=1)
    if abs(p.ref_tuning) > 1e-6:
        shift = int(round(p.ref_tuning)) % 12
        pcp = np.roll(pcp, -shift)
    templates = key_templates()
    scores = {k: _cosine(pcp, tmpl) for k, tmpl in templates.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_key, best_score = ranked[0]
    second_score = ranked[1][1]
    margin = max(0.0, best_score - second_score)
    conf = float(1 / (1 + np.exp(-(best_score * 4 + margin * 6))))
    best_major = max([(k, v) for k, v in scores.items() if k.endswith('major')],
key=lambda x: x[1])
    best_minor = max([(k, v) for k, v in scores.items() if k.endswith('minor')],
key=lambda x: x[1])
    return {"key": best_key, "confidence": conf, "best_major": {"key":
best_major[0], "score": best_major[1]},
            "best_minor": {"key": best_minor[0], "score": best_minor[1]},
"scores": scores, "pcp": pcp.tolist()}

KEYS24 = [f"{n} major" for n in PITCH_CLASS_NAMES] + [f"{n} minor" for n in PITCH_CLASS_NAMES]

def circle_of_fifths_distance(a_idx, b_idx):
    order = np.array([0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5])
    pos = {pc: i for i, pc in enumerate(order)}
    return min((pos[a_idx] - pos[b_idx]) % 12, (pos[b_idx] - pos[a_idx]) % 12)

def key_index_map():
    idx = {}
    for i, n in enumerate(PITCH_CLASS_NAMES):
        idx[f"{n} major"] = ("major", i)
    for i, n in enumerate(PITCH_CLASS_NAMES):
        idx[f"{n} minor"] = ("minor", i)
    return idx

def emission_scores(C_seq):
    tmpls = key_templates()
    K = len(KEYS24)
    T = C_seq.shape[1]
    em = np.zeros((K, T), dtype=float)
    pcp_frames = C_seq / (np.sum(C_seq, axis=0, keepdims=True) + 1e-9)
    for t in range(T):
        v = pcp_frames[:, t]
        v = v / (np.linalg.norm(v) + 1e-12)
        for k, key in enumerate(KEYS24):
            tmpl = tmpls[key]
            em[k, t] = float(np.dot(v, tmpl / (np.linalg.norm(tmpl) + 1e-12)))
    em = (em - em.mean(axis=0, keepdims=True)) / (em.std(axis=0, keepdims=True) + 1e-9)
    return em

def transition_matrix(lmbd=1.0, minor_switch_penalty=0.5):
    idx = key_index_map()
    Tmat = np.zeros((24, 24), dtype=float)
    for i, ki in enumerate(KEYS24):
        mode_i, pc_i = idx[ki]
        for j, kj in enumerate(KEYS24):
            mode_j, pc_j = idx[kj]
            d5 = circle_of_fifths_distance(pc_i, pc_j)
            cost = d5
            if mode_i != mode_j:
                rel = (pc_i - 3) % 12 if mode_i == 'major' else (pc_i + 3) % 12
                if pc_j != rel:
                    cost += minor_switch_penalty
            Tmat[i, j] = np.exp(-lmbd * cost)
        Tmat[i, :] /= (Tmat[i, :].sum() + 1e-12)
    return Tmat

def viterbi(emiss, trans):
    K, T = emiss.shape
    dp = np.zeros((K, T))
    back = np.zeros((K, T), dtype=int)
    dp[:, 0] = np.log(np.ones(K) / K + 1e-12) + emiss[:, 0]
    for t in range(1, T):
        prev = dp[:, t - 1][:, None] + np.log(trans + 1e-12)
        back[:, t] = np.argmax(prev, axis=0)
        dp[:, t] = np.max(prev, axis=0) + emiss[:, t]
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(dp[:, -1]))
    for t in range(T - 2, -1, -1):
        path[t] = back[path[t + 1], t + 1]
    return path

def decode_key_sequence(Cb_clean, lmbd=1.2):
    em = emission_scores(Cb_clean)
    trans = transition_matrix(lmbd=lmbd, minor_switch_penalty=0.5)
    path = viterbi(em, trans)
    return [KEYS24[i] for i in path]

def weighted_global_key(C_seq):
    tmpls = key_templates()
    agg = {k: 0.0 for k in tmpls}
    for t in range(C_seq.shape[1]):
        v = C_seq[:, t]
        v = v / (np.linalg.norm(v) + 1e-12)
        s = {k: float(np.dot(v, tmpls[k] / (np.linalg.norm(tmpls[k]) + 1e-12))) for k in tmpls}
        top = max(s.values())
        for k, val in s.items():
            agg[k] += top * val
    return max(agg.items(), key=lambda kv: kv[1])[0]

TRIADS = [f"{n}:maj" for n in PITCH_CLASS_NAMES] + [f"{n}:min" for n in PITCH_CLASS_NAMES]

def chord_templates() -> Dict[str, np.ndarray]:
    tmpl_major_root = np.zeros(12, dtype=float)
    tmpl_major_root[[0, 4, 7]] = 1.0
    tmpl_minor_root = np.zeros(12, dtype=float)
    tmpl_minor_root[[0, 3, 7]] = 1.0
    tmpls = {}
    for i, name in enumerate(PITCH_CLASS_NAMES):
        tmpls[f"{name}:maj"] = np.roll(tmpl_major_root, i)
        tmpls[f"{name}:min"] = np.roll(tmpl_minor_root, i)
    return tmpls

def score_chords_per_frame(Cb: np.ndarray) -> np.ndarray:
    # Numerically stable scoring to avoid warnings/overflows
    tmpls = chord_templates()
    mat = np.stack([tmpls[ch] for ch in TRIADS], axis=0)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    Cn = Cb / (np.linalg.norm(Cb, axis=0, keepdims=True) + 1e-12)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    Cn = np.nan_to_num(Cn, nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        out = mat @ Cn
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

def _median_smooth_labels(labels: np.ndarray, k: int = 3) -> np.ndarray:
    if k < 3 or k % 2 == 0 or labels.size < k:
        return labels
    pad = k // 2
    out = labels.copy()
    for i in range(labels.size):
        s = max(0, i - pad)
        e = min(labels.size, i + pad + 1)
        window = labels[s:e]
        out[i] = np.bincount(window).argmax()
    return out

def _collapse_segments(labels: np.ndarray, times: np.ndarray, min_dur: float
= 0.5):
    segs = []
    if labels.size == 0:
        return segs
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            s_time = times[start]
            e_time = times[i] if i < len(times) else times[-1]
            if e_time - s_time >= min_dur:
                segs.append((s_time, e_time, int(labels[start])))
            start = i
    return segs

def _fmt_mmss(t: float) -> str:
    if t < 0 or not np.isfinite(t):
        return "0:00"
    m = int(t // 60)
    s = int(t % 60)
    return f"{m}:{s:02d}"

def _pychord_name_from_notes(note_names: List[str]) -> Optional[str]:
    try:
        from pychord import find_chords_from_notes  # type: ignore
    except Exception:
        return None
    try:
        cand = find_chords_from_notes(note_names)
        if not cand:
            return None
        # Prefer simple triads/sevenths if available
        cand_sorted = sorted(cand, key=lambda c: (('maj7' in c or 'min7' in c or
'7' in c), len(c)))
        return cand_sorted[0]
    except Exception:
        return None

def detect_chords(Cb: np.ndarray,
                  beat_frames: np.ndarray,
                  sr: int,
                  hop_length: int,
                  smooth_k: int = 3,
                  min_seg_dur: float = 0.75,
                  kept_mask: np.ndarray = None,
                  use_pychord: bool = False):

    T_cb = Cb.shape[1]

    # Build time axis
    if beat_frames is None or len(beat_frames) == 0:
        frames = np.arange(T_cb)
        times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
    else:
        b = np.asarray(beat_frames, dtype=float)
        bt = librosa.frames_to_time(b, sr=sr, hop_length=hop_length)

        if T_cb == len(b) - 1:
            mids = 0.5 * (bt[:-1] + bt[1:])
            times = mids
        elif T_cb == len(b):
            times = bt
        else:
            x = np.arange(len(bt))
            xi = np.linspace(0, len(bt) - 1, T_cb)
            times = np.interp(xi, x, bt)

        if kept_mask is not None:
            kept_mask = np.asarray(kept_mask, dtype=bool)
            if times.shape[0] != kept_mask.shape[0]:
                x = np.arange(times.shape[0])
                xi = np.linspace(0, times.shape[0] - 1, kept_mask.shape[0])
                times_resized = np.interp(xi, x, times)
                times = times_resized
            times = times[kept_mask]

        if times.shape[0] != T_cb:
            if times.shape[0] > T_cb:
                times = times[:T_cb]
            else:
                pad = np.full(T_cb - times.shape[0], times[-1] if times.size
else 0.0)
                times = np.concatenate([times, pad], axis=0)

    # Score chords per frame
    S = score_chords_per_frame(Cb)
    raw = np.argmax(S, axis=0)
    lab = _median_smooth_labels(raw, k=smooth_k)
    seg_idx = _collapse_segments(lab, times, min_dur=min_seg_dur)

    chord_segs: List[Tuple[float, float, str, float]] = []
    for (s, e, lid) in seg_idx:
        idxs = np.where(lab == lid)[0]
        in_win = (times[idxs] >= s) & (times[idxs] < e)
        idxs = idxs[in_win]
        avg_score = float(np.mean(S[lid, idxs])) if idxs.size else float(np.mean(S[lid, :]))
        name = TRIADS[lid]  # e.g., "C:maj" or "A:min"
        if use_pychord:
            seg_cols = idxs if idxs.size else np.arange(Cb.shape[1])
            chroma_avg = np.mean(Cb[:, seg_cols], axis=1)
            if chroma_avg.max() > 0:
                thr = max(0.35 * float(chroma_avg.max()),
float(np.sort(chroma_avg)[-3]) * 0.6)
                pcs = np.where(chroma_avg >= thr)[0]
                note_names = [PITCH_CLASS_NAMES[i] for i in pcs]
                nm = _pychord_name_from_notes(note_names)
                if nm:
                    name = nm
        chord_segs.append((s, e, name, avg_score))

    return chord_segs

def normalize_chord_name(name: str) -> str:
    # Convert "C:maj" -> "C", "A:min" -> "Am"
    if ":maj" in name:
        return name.split(":maj")[0]
    if ":min" in name:
        return name.split(":min")[0] + "m"
    return name

# ----------------------------
# BPM Only (optional endpoint use)
# ----------------------------

def analyze_bpm_audio(path: str,
                      sr: int = 22050,
                      hop_length: int = 512,
                      engine: Literal['librosa', 'aubio'] = 'librosa') -> dict:
    try:
        y, sr = librosa.load(path, sr=sr, mono=True)
        if engine == 'aubio':
            aub = _bpm_from_aubio(path, sr, hop_length)
            if aub is not None:
                bpm, beat_times = aub
                conf = 0.0
                if len(beat_times) >= 3:
                    intervals = np.diff(np.asarray(beat_times))
                    tempos = 60.0 / np.maximum(intervals, 1e-6)
                    tmed = float(np.median(tempos))
                    tv = float(np.std(tempos) / (tmed + 1e-9))
                    conf = float(max(0.0, 1.0 - min(tv, 1.0)))
                return {"source": "audio", "engine": "aubio", "bpm": float(bpm),
"beat_times": [float(t) for t in beat_times], "confidence": float(conf)}

        # Librosa
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=oenv, sr=sr,
hop_length=hop_length, units='frames')
        tempo = float(np.asarray(tempo).squeeze())
        beat_times = librosa.frames_to_time(beat_frames, sr=sr,
hop_length=hop_length)

        if len(beat_times) >= 2:
            intervals = np.diff(beat_times)
            med = float(np.median(intervals))
            bpm_est = 60.0 / med if med > 0 else tempo
            tempos = 60.0 / np.maximum(intervals, 1e-6)
            tmed = float(np.median(tempos))
            tv = float(np.std(tempos) / (tmed + 1e-9))
            conf = float(max(0.0, 1.0 - min(tv, 1.0)))
        else:
            bpm_est = tempo
            conf = 0.0

        return {"source": "audio", "engine": "librosa", "bpm": float(bpm_est),
"beat_times": [float(t) for t in beat_times], "confidence": float(conf)}
    except Exception as e:
        return {"error": f"BPM analysis failed: {e}"}

# ----------------------------
# MIDI helpers (unchanged)
# ----------------------------

def analyze_midi_with_miditoolkit(path: str) -> dict:
    try:
        import miditoolkit  # type: ignore
    except Exception:
        raise RuntimeError("miditoolkit is not installed")

    try:
        midi = miditoolkit.midi.parser.MidiFile(path)
        ticks_per_beat = midi.ticks_per_beat or 480
        tempo = midi.tempo_changes[0].tempo if midi.tempo_changes else 120.0
        sec_per_tick = 60.0 / (tempo * ticks_per_beat)

        notes = []
        for inst in midi.instruments:
            for n in inst.notes:
                notes.append((n.start, n.end, n.pitch))
        if not notes:
            return {"error": "No notes in MIDI"}

        start_tick = min(n[0] for n in notes)
        end_tick = max(n[1] for n in notes)
        beat_size = ticks_per_beat  # 1 beat
        pc_hist_time: List[Tuple[float, float, np.ndarray]] = []
        t = start_tick
        while t < end_tick:
            t2 = t + beat_size
            pc = np.zeros(12, dtype=float)
            for s, e, pnum in notes:
                ov = max(0, min(e, t2) - max(s, t))
                if ov > 0:
                    pc[pnum % 12] += ov
            if pc.sum() > 0:
                pc = pc / (np.linalg.norm(pc) + 1e-12)
            pc_hist_time.append((t * sec_per_tick, t2 * sec_per_tick, pc))
            t = t2

        tmpls = key_templates()
        agg = np.zeros(12, dtype=float)
        for _, _, pc in pc_hist_time:
            agg += pc
        agg_n = agg / (np.linalg.norm(agg) + 1e-12)
        best_key, best_score = max(
            ((k, float(np.dot(agg_n, v / (np.linalg.norm(v) + 1e-12)))) for k, v
in tmpls.items()),
            key=lambda kv: kv[1]
        )

        chords = []
        try:
            from pychord import find_chords_from_notes  # type: ignore
        except Exception:
            find_chords_from_notes = None  # type: ignore

        for s, e, pc in pc_hist_time:
            name = None
            if find_chords_from_notes is not None:
                note_names = [PITCH_CLASS_NAMES[i] for i in np.where(pc >=
(pc.max() * 0.5 if pc.max() > 0 else 0))[0]]
                if note_names:
                    cands = find_chords_from_notes(note_names)
                    if cands:
                        name = cands[0]
            if not name:
                scores = (np.stack([chord_templates()[ch] for ch in TRIADS]) @
(pc / (np.linalg.norm(pc) + 1e-12)))
                name = TRIADS[int(np.argmax(scores))]
            chords.append({
                "start_mmss": _fmt_mmss(s),
                "end_mmss": _fmt_mmss(e),
                "start_seconds": float(s),
                "end_seconds": float(e),
                "chord": name,
                "avg_score": 0.0
            })

        # Conform to the same schema as audio
        chords_slim = [{
            "start_mmss": c["start_mmss"],
            "end_mmss": c["end_mmss"],
            "chord": c["chord"],
            "avg_score": c["avg_score"],
        } for c in chords]

        camelot = key_to_camelot(best_key) or ""
        neighbors = camelot_neighbors(camelot) if camelot else []
        neighbor_keys = [_CAMELOT_TO_KEY.get(c, "") for c in neighbors]

        return {
            "best_key": best_key,
            "bpm": int(round(float(tempo))),
            "confidence_score": float(best_score),
            "Camelot_key": camelot,
            "camelot_recommendations": {
                "codes": neighbors,
                "keys": neighbor_keys
            },
            "chords": chords_slim
        }
    except Exception as e:
        return {"error": f"MIDI analysis failed: {e}"}

def analyze_bpm_midi(path: str) -> dict:
    try:
        import miditoolkit  # type: ignore
    except Exception:
        return {"error": "miditoolkit is not installed"}
    try:
        midi = miditoolkit.midi.parser.MidiFile(path)
        if not midi.tempo_changes:
            return {"source": "midi", "engine": "midi", "bpm": 120.0,
"confidence": 0.0, "note": "No explicit tempo; defaulted to 120"}
        tempos = [float(tc.tempo) for tc in midi.tempo_changes]
        vals, counts = np.unique(np.round(tempos, 2), return_counts=True)
        bpm = float(vals[int(np.argmax(counts))])
        conf = float(min(1.0, np.max(counts) / max(1, len(tempos))))
        return {"source": "midi", "engine": "midi", "bpm": bpm, "confidence":
conf}
    except Exception as e:
        return {"error": f"MIDI BPM analysis failed: {e}"}

# ----------------------------
# Unified analysis
# ----------------------------

def detect_key_detailed(path: str, p: KeyDetectParams):
    y, sr = load_audio(path, p)
    Cb, beat_frames, beat_times, auto_ref, tempo_bpm = compute_chroma_beats(y,
sr, p, source_path=path)

    ptmp = deepcopy(p)
    ptmp.ref_tuning = auto_ref

    Cb_clean, kept = drop_low_energy_frames(Cb, thresh_quantile=0.2)
    if Cb_clean.shape[1] < 4:
        Cb_clean = Cb

    keys_seq = decode_key_sequence(Cb_clean, lmbd=1.2)

    wv_key = weighted_global_key(Cb_clean)
    res = estimate_key(Cb_clean, ptmp)

    chord_segs = detect_chords(
        Cb=Cb_clean,
        beat_frames=beat_frames,
        sr=sr,
        hop_length=p.hop_length,
        smooth_k=3,
        min_seg_dur=0.75,
        kept_mask=kept,
        use_pychord=p.use_pychord
    )

    # Build chord JSON in two shapes: legacy & structured
    chords_legacy = []
    chords_structured = []
    for (s, e, name, sc) in chord_segs:
        chords_legacy.append({
            "start_mmss": _fmt_mmss(s),
            "end_mmss": _fmt_mmss(e),
            "chord": name,
            "avg_score": float(sc)
        })
        chords_structured.append({
            "chord": normalize_chord_name(name),
            "startTime": float(s),
            "duration": float(max(0.0, e - s)),
            "confidence": float(sc),
            "notes": []
        })

    best_key = res["key"]  # e.g., "C major"
    parts = best_key.split(" ")
    key_root = parts[0] if parts else "C"
    key_mode = parts[1] if len(parts) > 1 else "major"
    scale_str = "Minor" if "min" in key_mode.lower() else "Major"

    # BPM confidence based on beat regularity (librosa-style)
    bpm_conf = 0.0
    if len(beat_times) >= 3:
        intervals = np.diff(np.asarray(beat_times))
        tempos = 60.0 / np.maximum(intervals, 1e-6)
        tmed = float(np.median(tempos))
        tv = float(np.std(tempos) / (tmed + 1e-9))
        bpm_conf = float(max(0.0, 1.0 - min(tv, 1.0)))

    # Robust BPM (more accurate, adjusted for half/double-time)
    bpm_val, bpm_conf = robust_bpm(y=y, sr=sr, hop_length=p.hop_length, beat_times=beat_times)

    # Camelot code + neighbor recommendations
    camelot = key_to_camelot(best_key) or ""
    neighbors = camelot_neighbors(camelot) if camelot else []
    neighbor_keys = [_CAMELOT_TO_KEY.get(c, "") for c in neighbors]

    # Final JSON: key + confidence_score + bpm + Camelot + chords
    return {
        "best_key": best_key,
        "bpm": int(bpm_val),
        "confidence_score": float(res["confidence"]),
        "Camelot_key": camelot,
        "camelot_recommendations": {
            "codes": neighbors,
            "keys": neighbor_keys
        },
        "chords": chords_legacy
    }

# ----------------------------
# Flask app
# ----------------------------

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_audio_endpoint():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file part in the request"}), 400
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        with tempfile.NamedTemporaryFile(delete=True,
suffix=os.path.splitext(file.filename)[1]) as temp_audio:
            file.save(temp_audio.name)
            try:
                # Optional flags
                analyzer = request.form.get('analyzer', 'key')  # 'key' (default) or 'bpm'
                beat_engine = request.form.get('beat_engine', 'librosa')
                use_pychord = request.form.get('use_pychord', 'false').lower() in ('1', 'true', 'yes', 'on')
                input_type = request.form.get('input_type', 'audio')  # 'audio' or 'midi'

                # BPM-only branch if requested
                if analyzer == 'bpm':
                    if input_type == 'midi' or temp_audio.name.lower().endswith(('.mid', '.midi')):
                        return jsonify(analyze_bpm_midi(temp_audio.name))
                    else:
                        res = analyze_bpm_audio(temp_audio.name, sr=22050, hop_length=512,
                                                engine='aubio' if beat_engine == 'aubio' else 'librosa')
                        return jsonify(res)

                # Default: unified chord + bpm analysis
                if input_type == 'midi' or temp_audio.name.lower().endswith(('.mid', '.midi')):
                    midi_res = analyze_midi_with_miditoolkit(temp_audio.name)
                    # Wrap MIDI into the unified envelope where possible
                    bpm_val = float(midi_res.get("bpm", 120.0))
                    chords = midi_res.get("chords", [])
                    key_str = midi_res.get("best_key", "C major")
                    key_root = key_str.split(" ")[0]
                    mode = (key_str.split(" ")[1] if len(key_str.split(" ")) > 1
else "major")
                    scale_str = "Minor" if "min" in mode.lower() else "Major"

                    chords_structured = []
                    for c in chords:
                        s = float(c.get("start_seconds", 0.0))
                        e = float(c.get("end_seconds", s))
                        chords_structured.append({
                            "chord": normalize_chord_name(c.get("chord", "N")),
                            "startTime": s,
                            "duration": max(0.0, e - s),
                            "confidence": float(c.get("avg_score", 0.8)),
                            "notes": []
                        })

                    return jsonify({
                        "bpm": bpm_val,
                        "bpm_confidence": float(midi_res.get("confidence",
0.0)),
                        "time_signature": "4/4",
                        "beat_times": [],
                        "chordProgression": {
                            "key": key_root,
                            "scale": scale_str,
                            "confidence": float(midi_res.get("score", 0.8)),
                            "chords": chords_structured
                        },
                        "best_key": key_str,
                        "score": float(midi_res.get("score", 0.8)),
                        "chords": chords  # legacy list with mm:ss
                    })

                params = KeyDetectParams(
                    sr=22050,
                    hop_length=4096,
                    chroma_type='cqt',
                    use_hpss=True,
                    beat_engine='aubio' if beat_engine == 'aubio' else
'librosa',
                    use_pychord=use_pychord
                )
                analysis_result = detect_key_detailed(temp_audio.name, params)
                return jsonify(analysis_result)

            except Exception as e:
                return jsonify({"error": f"An error occurred during analysis: {str(e)}"}), 500

    return jsonify({"error": "Unknown error"}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 to accept connections from your iPad on the same network
    app.run(host='0.0.0.0', port=5001, debug=True)
