import os
import time
import subprocess
import numpy as np
import jetson_utils_python as jetson_utils
import jetson_inference
import vlc
# === SCRIPT 1에서 병합된 모듈 START ===
from uploader import upload_file_to, CONT_FULL, CONT_CROP, SAS_FULL, SAS_CROP
from datetime import datetime, timezone, timedelta
import certifi
from pymongo import MongoClient
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError
# === SCRIPT 1에서 병합된 모듈 END ===

# === SCRIPT 1에서 병합된 DB 및 클라우드 설정 START ===
HOST = os.getenv("COSMOS_HOST")
USER = os.getenv("COSMOS_USER")
PASS = os.getenv("COSMOS_PASS")
DB_NAME = os.getenv("COSMOS_DB")
COL_NAME = os.getenv("COSMOS_COL")

URI = f"mongodb://{USER}:{PASS}@{HOST}:10255/?ssl=true&replicaSet=globaldb&retrywrites=false"
KST = timezone(timedelta(hours=9))

# 데이터베이스 클라이언트 초기화
client = MongoClient(
    URI,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=8000
)
db = client[DB_NAME]
col = db[COL_NAME]
# === SCRIPT 1에서 병합된 DB 및 클라우드 설정 END ===

# === 같은 컬렉션 안의 메타 문서로 시퀀스 관리 ===
SEQ_META_ID = "__seq_meta"   # intrusion_info 컬렉션에 이 _id의 문서가 '번호표' 역할을 함

def next_id_in_same_collection(width=4):
    """
    intrusion_info 컬렉션 안의 메타 문서(__seq_meta)의 seq 값을 원자적으로 +1하고,
    0-padding된 문자열(예: '0007')을 반환한다.
    문서가 없으면 upsert로 자동 생성된다.
    """
    doc = col.find_one_and_update(
        {"_id": SEQ_META_ID},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER
    )
    return str(doc["seq"]).zfill(width)


# === SCRIPT 1에서 병합된 이름 지정 유틸리티 START ===
def sanitize_label(s: str) -> str:
    """공백/대문자/특수문자를 정리하여 파일명으로 사용 가능하게 만듭니다."""
    s = s.strip().lower().replace(" ", "_")
    return "".join(ch for ch in s if (ch.isalnum() or ch in ("_", "-")))

def label_prefix(label: str, crop: bool) -> str:
    """업로드 파일명의 접두사를 생성합니다 (예: 'crop_dog', 'full_human')."""
    return f"{'crop' if crop else 'full'}_{label}"
# === SCRIPT 1에서 병합된 이름 지정 유틸리티 END ===


# =========================
# Box 변환 유틸
# =========================
def tlbr_to_xyah(tlbr):
    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]
    cx = tlbr[0] + w * 0.5
    cy = tlbr[1] + h * 0.5
    a = w / max(h, 1e-6)
    return np.array([cx, cy, a, h], dtype=np.float32)

def xyah_to_tlbr(xyah):
    cx, cy, a, h = xyah
    w = a * h
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return np.array([x1, y1, x2, y2], dtype=np.float32)

# ── 박스 스타일: 'outline' = 테두리, 'yolo' = 코너 강조
STYLE = "yolo"

def draw_rect_outline(img, x1,y1,x2,y2, color=(0,255,0,255)):
    jetson_utils.cudaDrawRect(img, (int(x1), int(y1), int(x2), int(y2)), color)

def draw_yolo_corners(img, x1,y1,x2,y2, color=(0,255,0,255), k=0.25, t=2):
    x1, y1, x2, y2 = map(int, (x1,y1,x2,y2))
    w, h = x2 - x1, y2 - y1
    L = int(min(w, h) * k)
    for i in range(t):
        jetson_utils.cudaDrawLine(img, (x1, y1+i),   (x1+L, y1+i),   color)
        jetson_utils.cudaDrawLine(img, (x1+i, y1),   (x1+i, y1+L),   color)
        jetson_utils.cudaDrawLine(img, (x2-L, y1+i), (x2,   y1+i),   color)
        jetson_utils.cudaDrawLine(img, (x2-1-i, y1), (x2-1-i, y1+L), color)
        jetson_utils.cudaDrawLine(img, (x1, y2-1-i), (x1+L, y2-1-i), color)
        jetson_utils.cudaDrawLine(img, (x1+i, y2-L), (x1+i, y2),     color)
        jetson_utils.cudaDrawLine(img, (x2-L, y2-1-i), (x2,   y2-1-i), color)
        jetson_utils.cudaDrawLine(img, (x2-1-i, y2-L), (x2-1-i, y2),   color)

def draw_box(img, x1,y1,x2,y2, color=(0,255,0,255)):
    if STYLE == "outline":
        draw_rect_outline(img, x1,y1,x2,y2, color)
    else:
        draw_yolo_corners(img, x1,y1,x2,y2, color)

def id2color(tid):
    np.random.seed(tid)
    r,g,b = np.random.randint(80, 255, 3).tolist()
    return (int(r), int(g), int(b), 255)

def class2color(cid):
    np.random.seed(cid + 12345)
    r,g,b = np.random.randint(60, 220, 3).tolist()
    return (int(r), int(g), int(b), 220)

# =========================
# IoU & Mahalanobis
# =========================
def iou_xyxy(a, b):
    N, M = a.shape[0], b.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])
    area_b = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)

def mahalanobis_squared(y, S_inv):
    return np.einsum("ki,ij,kj->k", y, S_inv, y)

# =========================
# Kalman Filter
# =========================
class KalmanFilterXYAH:
    def __init__(self, dt=1.0):
        self.dt = dt
        self._F = np.eye(8, dtype=np.float32)
        for i in range(4): self._F[i, i+4] = dt
        self._H = np.zeros((4, 8), dtype=np.float32)
        self._H[0,0] = self._H[1,1] = self._H[2,2] = self._H[3,3] = 1.0
        q_pos, q_vel = 1.0, 10.0
        self._Q = np.diag([q_pos, q_pos, 1e-2, q_pos, q_vel, q_vel, 1e-3, q_vel]).astype(np.float32)
        self._R = np.diag([1.0, 1.0, 1e-2, 1.0]).astype(np.float32)
        self._I8 = np.eye(8, dtype=np.float32)

    def initiate(self, xyah):
        mean = np.zeros((8,), dtype=np.float32); mean[:4] = xyah
        P = np.diag([10, 10, 1e-1, 10, 100, 100, 1e-2, 100]).astype(np.float32)
        return mean, P

    def predict(self, mean, P):
        mean = np.dot(self._F, mean)
        P = np.dot(np.dot(self._F, P), self._F.T) + self._Q
        return mean, P

    def project(self, mean, P):
        S = np.dot(np.dot(self._H, P), self._H.T) + self._R
        z = np.dot(self._H, mean)
        return z, S

    def update(self, mean, P, z_obs):
        z_pred, S = self.project(mean, P)
        HP = np.dot(self._H, P)
        K  = np.linalg.solve(S, HP).T
        mean = mean + np.dot(K, (z_obs - z_pred))
        KH = np.dot(K, self._H)
        P  = np.dot((self._I8 - KH), P)
        return mean, P

# =========================
# Track / BYTETracker
# =========================
class TrackKF:
    __slots__ = ("mean","cov","score","id","age","miss","hit","class_id","activated","kf")
    def __init__(self, xyah, score, class_id, tid, kf: KalmanFilterXYAH):
        self.kf = kf; self.mean, self.cov = kf.initiate(xyah)
        self.score = float(score); self.class_id = int(class_id); self.id = tid
        self.age = 0; self.miss = 0; self.hit = 1; self.activated = True

    @property
    def tlbr(self): return xyah_to_tlbr(self.mean[:4])
    def predict(self): self.mean, self.cov = self.kf.predict(self.mean, self.cov); self.age += 1; self.miss += 1
    def update(self, tlbr, score, class_id=None):
        xyah = tlbr_to_xyah(tlbr)
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyah)
        self.score = float(score)
        if class_id is not None: self.class_id = int(class_id)
        self.hit += 1; self.miss = 0; self.activated = True

class BYTETrackerKF:
    def __init__(self, track_thresh=0.45, match_thresh=0.55, match_thresh_low=0.45,
                 buffer_ttl=30, min_box_area=10, gate_maha=True, maha_th=25.0,
                 dt=1.0, max_ids=1<<30, class_consistent=True):
        self.track_thresh, self.match_thresh, self.match_thresh_low = track_thresh, match_thresh, match_thresh_low
        self.buffer_ttl, self.min_box_area = buffer_ttl, min_box_area
        self.gate_maha, self.maha_th = gate_maha, maha_th
        self.kf = KalmanFilterXYAH(dt=dt)
        self.tracks = []; self._next_id = 1; self._max_ids = max_ids; self.class_consistent = class_consistent

    def _new_id(self): tid = self._next_id; self._next_id += 1; self._next_id = 1 if self._next_id >= self._max_ids else self._next_id; return tid
    @staticmethod
    def _greedy_match(iou_mat, iou_th):
        matches, u_a, u_b = [], list(range(iou_mat.shape[0])), list(range(iou_mat.shape[1]))
        if iou_mat.size == 0: return matches, u_a, u_b
        iou_copy = iou_mat.copy()
        while True:
            maxv = iou_copy.max() if iou_copy.size else 0.0
            if maxv < iou_th or maxv <= 0: break
            i, j = np.unravel_index(np.argmax(iou_copy), iou_copy.shape)
            matches.append((int(i), int(j))); iou_copy[i, :], iou_copy[:, j] = -1.0, -1.0
        matched_a = {m[0] for m in matches}; matched_b = {m[1] for m in matches}
        u_a = [i for i in range(iou_mat.shape[0]) if i not in matched_a]
        u_b = [j for j in range(iou_mat.shape[1]) if j not in matched_b]
        return matches, u_a, u_b

    def _gating_mask(self, dets_xyah, tracks):
        if not self.gate_maha or not tracks or not dets_xyah.size: return np.ones((len(tracks), len(dets_xyah)), dtype=bool)
        mask = np.zeros((len(tracks), len(dets_xyah)), dtype=bool)
        for i, t in enumerate(tracks):
            z_pred, S = self.kf.project(t.mean, t.cov); S_inv = np.linalg.inv(S)
            y = dets_xyah - z_pred[None, :]; d2 = mahalanobis_squared(y, S_inv); mask[i] = d2 < self.maha_th
        return mask

    def update(self, dets_tlbr_scores, class_ids=None):
        for t in self.tracks: t.predict()
        if dets_tlbr_scores is None or len(dets_tlbr_scores) == 0:
            self.tracks = [t for t in self.tracks if t.miss <= self.buffer_ttl]
            return [t for t in self.tracks if t.miss == 0 and t.activated]
        dets = np.asarray(dets_tlbr_scores, dtype=np.float32)
        cls_arr = np.zeros((dets.shape[0],), dtype=np.int32) if class_ids is None else np.asarray(class_ids, dtype=np.int32)
        high_mask = dets[:, 4] >= self.track_thresh
        dets_high, dets_low = dets[high_mask], dets[~high_mask]
        cls_high, cls_low  = cls_arr[high_mask], cls_arr[~high_mask]
        active_idx = list(range(len(self.tracks)))
        tr_tlbr = np.array([self.tracks[i].tlbr for i in active_idx], dtype=np.float32) if active_idx else np.zeros((0,4), np.float32)
        def _class_mask(tracks, det_classes):
            if not self.class_consistent or not tracks or not det_classes.size: return np.ones((len(tracks), len(det_classes)), dtype=bool)
            T, D = len(tracks), len(det_classes); m = np.zeros((T, D), dtype=bool)
            for ti, t in enumerate(tracks):
                for di in range(D): m[ti, di] = (t.class_id == int(det_classes[di]))
            return m
        iou_mat = iou_xyxy(tr_tlbr, dets_high[:, :4]) if dets_high.size else np.zeros((tr_tlbr.shape[0], 0), np.float32)
        if dets_high.size and self.tracks:
            dets_high_xyah = np.stack([tlbr_to_xyah(b) for b in dets_high[:, :4]], axis=0)
            gate = self._gating_mask(dets_high_xyah, [self.tracks[i] for i in active_idx])
            cmask = _class_mask([self.tracks[i] for i in active_idx], cls_high)
            iou_mat = np.where(gate & cmask, iou_mat, -1.0)
        matches, u_tr, u_dt = self._greedy_match(iou_mat, self.match_thresh)
        for (ti, di) in matches: self.tracks[active_idx[ti]].update(dets_high[di, :4], dets_high[di, 4], class_id=int(cls_high[di]))
        if len(u_tr) and len(dets_low):
            tr_rest = np.array([self.tracks[active_idx[i]].tlbr for i in u_tr], dtype=np.float32)
            iou_low = iou_xyxy(tr_rest, dets_low[:, :4])
            if self.tracks:
                dets_low_xyah = np.stack([tlbr_to_xyah(b) for b in dets_low[:, :4]], axis=0)
                gate2 = self._gating_mask(dets_low_xyah, [self.tracks[active_idx[i]] for i in u_tr])
                cmask2 = _class_mask([self.tracks[active_idx[i]] for i in u_tr], cls_low)
                iou_low = np.where(gate2 & cmask2, iou_low, -1.0)
            matches2, u_tr2, u_dt2 = self._greedy_match(iou_low, self.match_thresh_low)
            for (ti2, di2) in matches2: self.tracks[active_idx[u_tr[ti2]]].update(dets_low[di2, :4], dets_low[di2, 4], class_id=int(cls_low[di2]))
            u_tr_final, u_dt_high_final, u_dt_low_final = [u_tr[i] for i in u_tr2], [i for i in range(len(dets_high)) if i not in u_dt], u_dt2
        else: u_tr_final, u_dt_high_final, u_dt_low_final = u_tr, u_dt, list(range(len(dets_low)))
        for di in u_dt_high_final:
            tlbr = dets_high[di, :4]
            if (tlbr[2]-tlbr[0])*(tlbr[3]-tlbr[1]) >= self.min_box_area: self.tracks.append(TrackKF(tlbr_to_xyah(tlbr), dets_high[di, 4], int(cls_high[di]), self._new_id(), self.kf))
        self.tracks = [t for t in self.tracks if t.miss <= self.buffer_ttl]
        return [t for t in self.tracks if t.miss == 0 and t.activated]

# === SCRIPT 1에서 병합된 오디오 로직 START ===
# Script 2의 단일 오디오 로직을 대체하여 클래스별 오디오 및 쿨다운을 지원합니다.
vlc_instance = vlc.Instance('--no-xlib', '--no-video', '--aout=pulse')
player = vlc_instance.media_player_new()

def _get_media_duration_sec(path: str, timeout_sec: float = 2.0) -> float:
    try:
        media = vlc_instance.media_new(path)
        media.parse()
        dur_ms = media.get_duration()
        if dur_ms <= 0:
            start = time.time()
            while dur_ms <= 0 and (time.time() - start) < timeout_sec:
                time.sleep(0.05)
                dur_ms = media.get_duration()
        return max(0.1, dur_ms / 1000.0) if dur_ms > 0 else 3.2
    except Exception:
        return 3.2

def play_mp3_once(path: str, volume: int = 100) -> float:
    assert os.path.exists(path), f"MP3 not found: {path}"
    try:
        length_sec = _get_media_duration_sec(path)
        media = vlc_instance.media_new(path)
        player.set_media(media)
        try:
            player.audio_set_volume(int(volume))
        except Exception: pass
        player.stop(); player.play()
        print(f"[AUDIO] play: {os.path.basename(path)} ({length_sec:.2f}s)")
        return length_sec
    except Exception as e:
        print(f"[AUDIO] play failed: {e}"); return 0.0

def stop_audio():
    try: player.stop()
    except Exception: pass
# === SCRIPT 1에서 병합된 오디오 로직 END ===


# ==== I/O / Labels ====
WIDTH, HEIGHT = 640, 480
LABELS_PATH = "/home/jetson/vigil/labels.txt"

def normalize(s):
    return " ".join(s.strip().lower().replace("_", " ").split())

id_to_label = []
label_to_id = {}
with open(LABELS_PATH, "r") as f:
    for idx, line in enumerate(f):
        name = line.strip()
        if name: id_to_label.append(name); label_to_id[normalize(name)] = idx

# ====== 클래스별 운용 튜닝 ======
CLASS_CONF_TH = { "Human": 0.35, "Wild Boar": 0.40, "Wild Rabbit": 0.40, "Bird": 0.35, "Siberian Chipmunk": 0.35, "Squirrel": 0.45, "Weasel": 0.45, "Leopard Cat": 0.40, "Racoon": 0.50, "Water Deer": 0.40, "Dog": 0.50, }
DEFAULT_CONF_TH = 0.50
CLASS_MIN_AREA = { "bird": 64, "siberian chipmunk": 64, "squirrel": 100, "human": 144, "dog": 144, }
DEFAULT_MIN_AREA = 144

# ====== 위험도 기반 운용 정책 ======
RISK_LEVEL = { "Human": "high", "Wild Boar": "high", "Dog": "high", "Water Deer": "high", "Racoon": "high", "Leopard Cat": "high", "Squirrel": "high", "Weasel": "high", "Wild Rabbit": "high", "Bird": "high" }
RISK_POLICY = { "high": {"persist": 1.5}, "medium": {"persist": 2.5}, "low": {"persist": 3.0} }
def risk_level_of(name_norm: str) -> str: return RISK_LEVEL.get(name_norm, "low")
def persist_sec_of(name_norm: str) -> float: return RISK_POLICY[risk_level_of(name_norm)]["persist"]

MONITOR_CLASSES = None   # e.g., {"human", "dog", "wild boar"}

# =========== Camera / Display / Model ===========
camera = jetson_utils.gstCamera(WIDTH, HEIGHT, "/dev/video0")
# display_local = jetson_utils.videoOutput()
display_rtmp  = jetson_utils.videoOutput("rtmp://agrilook-be-stream.koreacentral.cloudapp.azure.com:1935/live/jetson1")

net = jetson_inference.detectNet(argv=["--model=/home/jetson/vigil/mb2-ssd-lite_320.onnx", f"--labels={LABELS_PATH}", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes", "--threshold=0.25", "--verbose"])
tracker = BYTETrackerKF(track_thresh=0.40, match_thresh=0.45, match_thresh_low=0.25, buffer_ttl=90, min_box_area=DEFAULT_MIN_AREA, gate_maha=True, maha_th=60.0, dt=1.0, class_consistent=False)
font = jetson_utils.cudaFont()

# === SCRIPT 1에서 병합된 캡처 및 오디오 상태 관리 START ===
CAPTURE_DIR = "/home/jetson/vigil/captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_FILES = {
    # 키는 sanitize_label()을 통과한 형태여야 함
    "human": os.path.join(BASE_DIR, "human.mp3"),
    "dog":   os.path.join(BASE_DIR, "dog.mp3"),
    "wild_boar": os.path.join(BASE_DIR, "wild_boar.mp3")
}
AUDIO_COOLDOWN_SEC = 5.0
last_audio_ts_class = {}   # {Label: last_ts}
audio_playing  = False
audio_started  = 0.0
audio_len_sec  = 0.0
# === SCRIPT 1에서 병합된 캡처 및 오디오 상태 관리 END ===


seen_since = {}
captured_ids = set()
start = time.time()


# ===== 라벨(텍스트) 그리기 유틸 =====
def draw_label_with_bg(img, w, h, x, y, text, bg_color=(0,0,0,180), pad=3):
    try: tw, th = font.GetTextSize(text)
    except Exception: th, tw = 18, 8 * len(text)
    x1, y1 = int(max(0, x)), int(max(0, y))
    x2, y2 = int(min(w-1, x1 + tw + pad*2)), int(min(h-1, y1 + th + pad*2))
    jetson_utils.cudaDrawRect(img, (x1, y1, x2, y2), bg_color)
    font.OverlayText(img, w, h, text, x1 + pad, y1 + pad, font.White, font.Gray40)

# === SCRIPT 1에서 병합된 캡처/업로드 함수 START ===
def crop_save_and_upload(img, dt_str, bbox_xyxy, prefix="intrusion", persist_dir=CAPTURE_DIR):
    os.makedirs(persist_dir, exist_ok=True)
    x1, y1, x2, y2 = bbox_xyxy
    x1, y1 = max(0, int(x1)), max(0, int(y1)); x2, y2 = int(x2), int(y2)
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    try:
        fmt = getattr(img, 'format', 'rgba32f')
        crop_gpu = jetson_utils.cudaAllocMapped(width=w, height=h, format=fmt)
        jetson_utils.cudaCrop(img, crop_gpu, (x1, y1, x2, y2))
    except Exception as e:
        print(f"[WARN] cudaCrop failed, fallback to full frame. err={e}"); crop_gpu = img
    # ts = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{dt_str}.jpg"; local_path = os.path.join(persist_dir, filename)
    jetson_utils.saveImage(local_path, crop_gpu)
    try:
        blob_url = upload_file_to(local_path, filename, container=CONT_CROP, sas_url=SAS_CROP)
        print(f"[BLOB] uploaded: {blob_url}")
    except Exception as e: print(f"[BLOB] upload failed: {e}"); blob_url = ""
    return local_path, blob_url

def full_save_and_upload(img, dt_str, prefix="full", persist_dir=CAPTURE_DIR):
    os.makedirs(persist_dir, exist_ok=True)
    # ts = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{dt_str}.jpg"; local_path = os.path.join(persist_dir, filename)
    jetson_utils.saveImage(local_path, img)
    try:
        blob_url = upload_file_to(local_path, filename, container=CONT_FULL, sas_url=SAS_FULL)
        print(f"[BLOB-FULL] uploaded: {blob_url}")
    except Exception as e: print(f"[BLOB-FULL] upload failed: {e}"); blob_url = ""
    return local_path, blob_url
# === SCRIPT 1에서 병합된 캡처/업로드 함수 END ===


def get_class_conf_th(name_norm): return CLASS_CONF_TH.get(name_norm, DEFAULT_CONF_TH)
def get_min_area(name_norm): return CLASS_MIN_AREA.get(name_norm, DEFAULT_MIN_AREA)

print("=== Starting ==="); print("DISPLAY:", os.environ.get("DISPLAY")); print("XDG_SESSION_TYPE:", os.environ.get("XDG_SESSION_TYPE"))

try:
    while True:
        img, w, h = camera.CaptureRGBA()
        if img is None: print("Capture failed"); break

        # 1) Detect
        dets = net.Detect(img, w, h, overlay='none')

        # 2) 클래스별 임계값/면적 필터링
        det_list, cls_list = [], []
        for d in dets:
            cid = int(d.ClassID)
            if not (0 <= cid < len(id_to_label)): continue
            cname = id_to_label[cid]; cname_norm = normalize(cname)
            if MONITOR_CLASSES and cname_norm not in {normalize(x) for x in MONITOR_CLASSES}: continue
            if float(d.Confidence) < get_class_conf_th(cname_norm): continue
            x1, y1, x2, y2 = float(d.Left), float(d.Top), float(d.Right), float(d.Bottom)
            if (x2 - x1) * (y2 - y1) < get_min_area(cname_norm): continue
            det_list.append([x1, y1, x2, y2, float(d.Confidence)]); cls_list.append(cid)
        det_arr = np.array(det_list, dtype=np.float32) if det_list else np.zeros((0,5), np.float32)

        # 3) Tracker Update
        tracks = tracker.update(det_arr, class_ids=np.array(cls_list) if cls_list else None)

        # 4) 렌더링 / 캡처 / DB 저장
        now = time.time()
        active_ids = set()
        for t in tracks:
            tid = t.id; active_ids.add(tid)
            x1, y1, x2, y2 = t.tlbr
            if tid not in seen_since: seen_since[tid] = now
            elapsed = now - seen_since[tid]
            cname = id_to_label[t.class_id] if 0 <= t.class_id < len(id_to_label) else str(t.class_id)
            cname_norm = normalize(cname)
            draw_box(img, x1, y1, x2, y2, id2color(tid))
            label_bg = class2color(t.class_id if 0 <= t.class_id < len(id_to_label) else 0)
            label_text = f"{cname} | id:{tid} | s:{t.score:.2f} | t:{elapsed:.1f}"
            tx, ty = int(max(0, x1)), int(y1 - 22) if y1 >= 22 else int(y1 + 2)
            draw_label_with_bg(img, w, h, tx, ty, label_text, bg_color=label_bg)

            # 위험도 기반 지속 시간 체크 후 캡처/업로드/DB저장
            need_sec = persist_sec_of(cname_norm)
            if elapsed >= need_sec and tid not in captured_ids:
                # === SCRIPT 1에서 병합된 캡처/업로드/DB 저장 로직 START ===
                label = sanitize_label(cname)
                # 사람이 읽기 쉬운 시간 문자열
                # "YYYYMMDD-HHMMSS" KST(UTC+9)로 저장
                kst = timezone(timedelta(hours=9))
                dt_str = datetime.now(kst).strftime("%Y%m%d-%H%M%S")
                crop_local, crop_url = crop_save_and_upload(img, dt_str, (x1, y1, x2, y2), prefix=label_prefix(label, crop=True))
                full_local, full_url = full_save_and_upload(img, dt_str, prefix=label_prefix(label, crop=False))
                print(f"[CAPTURE] crop: {crop_local} | {crop_url}")
                print(f"[CAPTURE] full: {full_local} | {full_url}")
                # === DB로 class + datetime(문자열) 저장, _id는 시퀀스(0001...) ===
                try:
                    # 1) 새 번호 뽑기 (원자적 증가)
                    seq_str = next_id_in_same_collection(width=4)  # 0001, 0002, ...

                    conf_pct = int(round(t.score * 100))
                    conf_str = "{}%".format(conf_pct)

                    # 3) DB 저장 (요청대로 class와 datetime만)
                    col.insert_one({
                        "_id": seq_str,
                        "class": label,
                        "confidence": conf_str,
                        "datetime": dt_str
                    })
                    print(f"[DB] inserted _id={seq_str} class={label} conf={conf_str} datetime={dt_str}")

                except DuplicateKeyError:
                    # 동시 삽입으로 같은 _id가 생기는 아주 드문 경우: 다시 한 번만 재시도
                    try:
                        seq_str = next_id_in_same_collection(width=4)
                        dt_str = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                        col.insert_one({"_id": seq_str, "class": label, "datetime": dt_str})
                        print(f"[DB] inserted (retry) _id={seq_str} class={label} datetime={dt_str}")
                    except Exception as e2:
                        print(f"[DB] insert retry failed: {e2}")
                except Exception as e:
                    print(f"[DB] insert failed: {e}")

                # === SCRIPT 1에서 병합된 캡처/업로드/DB 저장 로직 END ===
                captured_ids.add(tid)

        for tid in list(seen_since.keys()):
            if tid not in active_ids: del seen_since[tid]

              # === SCRIPT 1에서 병합된 클래스별 오디오 재생 로직 START ===
        # Script 2의 단순 알람 로직을 대체합니다.
        classes_ready = set()
        for t in tracks:
            if t.id not in seen_since: continue

            # ▼▼▼▼▼▼▼▼▼▼▼ 이 부분이 수정되었습니다 ▼▼▼▼▼▼▼▼▼▼▼
            cname = id_to_label[t.class_id] if 0 <= t.class_id < len(id_to_label) else str(t.class_id)
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            cname_norm_policy = normalize(cname) # 위험도 정책 조회용
            cname_norm_audio = sanitize_label(cname) # 오디오 파일 키 조회용

            need_sec = persist_sec_of(cname_norm_policy)
            elapsed = now - seen_since[t.id]

            if elapsed >= need_sec and cname_norm_audio in AUDIO_FILES and os.path.exists(AUDIO_FILES[cname_norm_audio]):
                classes_ready.add(cname_norm_audio)

        play_order = list(AUDIO_FILES.keys())
        chosen_label = next((lbl for lbl in play_order if lbl in classes_ready), None)

        if chosen_label:
            last_ts = last_audio_ts_class.get(chosen_label, 0.0)
            if (not audio_playing) and (now - last_ts) >= AUDIO_COOLDOWN_SEC:
                mp3_file = AUDIO_FILES[chosen_label]
                audio_len_sec = play_mp3_once(mp3_file, volume=100)
                audio_started = time.time()
                audio_playing = True
                last_audio_ts_class[chosen_label] = now

        if audio_playing and (time.time() - audio_started) >= (audio_len_sec or 3.2):
            stop_audio()
            audio_playing = False
        # === SCRIPT 1에서 병합된 클래스별 오디오 재생 로직 END ===

        # 6) 출력
        # if display_local and display_local.IsStreaming():
        #     display_local.Render(img)
        #     display_local.SetStatus(f"{net.GetNetworkFPS():.0f} FPS | det_raw={len(dets)} | det_used={len(det_list)} | tracks={len(tracks)}")
        if display_rtmp:
            display_rtmp.Render(img)

        if time.time() - start > 120: # Timeout 설정 (필요시 조정)
            print("Timeout exit"); break

except KeyboardInterrupt: print("Interrupted")
finally:
    # 모든 리소스 정리
    for resource in [camera, display_rtmp]: # display_local
        if 'resource' in locals() and resource is not None:
            try: resource.Close()
            except Exception: pass
    try: player.stop()
    except Exception: pass
    print("Done.")