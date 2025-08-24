import os
import time
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
from collections import defaultdict
# === SCRIPT 1에서 병합된 모듈 END ===
# 맨 위 import 들 아래에 추가
try:
    from pydub import AudioSegment
except Exception as e:
    AudioSegment = None
    print("[AUDIO] pydub not available -> gain boost disabled:", e)

# 사운드 키별 증폭량(dB) — 원하면 숫자 바꾸세요
SOUND_GAIN_DB = {
    "voice": 12,   # 사람용
    "alarm": 12,   # 개(알람)
    "dog":   12,   # 새 -> 개 소리
    "wolf":  12,   # 다람쥐/라쿤/다람쥐(시베리아) 등
    "tiger": 12,   # 멧돼지/삵/고라니/족제비/토끼 등
}

# === (추가) 안전 증폭 유틸 ===
def safe_boost_db(src_path: str, desired_db: float) -> float:
    """
    파일의 헤드룸(-max_dBFS)을 넘지 않는 선에서 증폭 dB를 제한.
    pydub이 없거나 오류면 0(증폭 안 함) 반환.
    """
    if AudioSegment is None or desired_db <= 0:
        return 0.0
    try:
        audio = AudioSegment.from_file(src_path)
        headroom = max(0.0, -audio.max_dBFS)  # ex) -(-6.2)=6.2 -> 6.2dB까지 안전
        return min(desired_db, headroom)
    except Exception as e:
        print(f"[AUDIO] safe_boost_db error for {src_path}: {e}")
        return 0.0

def make_boosted_copy(src_path: str, gain_db: float, out_dir: str = "/tmp/boosted_sounds") -> str:
    """
    src_path를 gain_db만큼 증폭해서 out_dir에 mp3로 저장하고 그 경로를 반환.
    gain_db<=0이면 원본 경로 반환.
    pydub/ffmpeg이 없거나 실패하면 원본 경로 반환.
    """
    if AudioSegment is None or gain_db <= 0:
        return src_path
    try:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(src_path))[0]
        tag  = str(gain_db).replace('.', 'p').replace('-', 'm')  # 10.5 -> 10p5
        out_path = os.path.join(out_dir, f"{base}_boost{tag}dB.mp3")

        # 이미 만들어둔 게 있으면 재사용
        if os.path.exists(out_path):
            return out_path

        audio = AudioSegment.from_file(src_path)
        boosted = audio + gain_db
        boosted.export(out_path, format="mp3")  # ffmpeg 필요
        print(f"[AUDIO] made boosted copy: {out_path} (+{gain_db} dB)")
        return out_path
    except Exception as e:
        print(f"[AUDIO] make_boosted_copy error for {src_path}: {e}")
        return src_path

# === SCRIPT 1에서 병합된 DB 및 클라우드 설정 START ===
HOST = os.getenv("COSMOS_HOST")
USER = os.getenv("COSMOS_USER")
PASS = os.getenv("COSMOS_PASS")
DB_NAME = os.getenv("COSMOS_DB")
COL_NAME = os.getenv("COSMOS_COL")

URI = f"mongodb://{USER}:{PASS}@{HOST}:10255/?ssl=true&replicaSet=globaldb&retrywrites=false"
KST = timezone(timedelta(hours=9))
FARM_ID = "farm_0001"

# --- 안전 DB 초기화: 연결 안 되면 스킵 모드 ---
client = None
db = None
col = None
db_ok = False   # ← 연결 상태 플래그

try:
    client = MongoClient(
        URI,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=3000,  # 짧은 타임아웃
        connectTimeoutMS=3000,
        socketTimeoutMS=3000,
    )
    # 실제 연결 확인
    client.admin.command("ping")
    db  = client[DB_NAME]
    col = db[COL_NAME]
    db_ok = True
    print("[DB] connected")
except Exception as e:
    print(f"[DB] unavailable -> skip DB writes: {e}")
    # db_ok=False 유지, col=None 유지 (저장 시 스킵)
DB_SKIP_LOG_THROTTLE = 10.0   # 같은 경고를 최소 10초 간격으로만 출력
last_db_skip_log = 0.0



# === SCRIPT 1에서 병합된 DB 및 클라우드 설정 END ===

# === 같은 컬렉션 안의 메타 문서로 시퀀스 관리 ===
SEQ_META_ID = "__seq_meta"   # intrusion_info 컬렉션에 이 _id의 문서가 '번호표' 역할을 함

def next_id_in_same_collection(width=4):
    """
    intrusion_info 컬렉션 안의 메타 문서(__seq_meta)의 seq 값을 원자적으로 +1하고,
    0-padding된 문자열(예: '0007')을 반환한다.
    문서가 없으면 upsert로 자동 생성된다.
    """
    if col is None:
        raise RuntimeError("DB not available")  # ← DB 없으면 호출 금지
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

def normalize(s):
    return " ".join(s.strip().lower().replace("_", " ").split())
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
        return length_sec
    except Exception as e:
        print(f"[AUDIO] play failed: {e}"); return 0.0

def stop_audio():
    try: player.stop()
    except Exception: pass
# === SCRIPT 1에서 병합된 오디오 로직 END ===

# === PoseNet 옵션 ===
ENABLE_POSENET = True
POSENET_PER_HUMAN_EVERY_N_FRAMES = 1   # 1=매프레임, 2=격프레임 등
POSE_OVERLAY_IN_BOX = "links,keypoints"  # crop 위에 그릴 오버레이 (links,keypoints,boxes 조합 가능)
POSE_PADDING = 6                        # bbox 주위 여유 픽셀

# === PoseNet 표시/제스처 설정 ===
# COCO-18 기준 keypoint ID (jetson-inference resnet18-body)
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
FACE_IDS = {0, 1, 2, 3, 4}
POSE_KP_TH = 0.15  # 키포인트 신뢰도 임계값

# 얼굴 제외한 링크(스켈레톤)만 그리기 위한 에지 정의
LIMB_EDGES = [
    (5, 6),           # shoulders
    (11, 12),         # hips
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11),          # left torso side
    (6, 12),          # right torso side
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]


# OK 제스처 스티키(부여 후 유지 시간)
OK_MEMORY_HANG_SEC = 30.0   # 트랙이 안 보이더라도 OK 상태를 이 시간(초)만큼 유지
track_last_seen = {}        # {track_id: 마지막으로 화면에 등장한 시각}

# ==== I/O / Labels ====
WIDTH, HEIGHT = 640, 480
LABELS_PATH = "/home/jetson/vigil/labels.txt"

id_to_label = []
label_to_id = {}

with open(LABELS_PATH, "r") as f:
    for idx, line in enumerate(f):
        name = line.strip()
        if name:
            id_to_label.append(name)
            label_to_id[normalize(name)] = idx

# 사람 클래스 ID
HUMAN_ID = label_to_id.get(normalize("Human"), None)
print("[POSE] HUMAN_ID:", HUMAN_ID)
# ====== 클래스별 운용 튜닝 ======
CLASS_CONF_TH = { "Human": 0.30, "Wild Boar": 0.40, "Wild Rabbit": 0.40, "Bird": 0.35, "Siberian Chipmunk": 0.35, "Squirrel": 0.45, "Weasel": 0.45, "Leopard Cat": 0.40, "Racoon": 0.50, "Water Deer": 0.40, "Dog": 0.50, }
DEFAULT_CONF_TH = 0.50
CLASS_MIN_AREA = { "bird": 64, "siberian chipmunk": 64, "squirrel": 100, "human": 144, "dog": 144, }
DEFAULT_MIN_AREA = 144

# ====== 위험도 기반 운용 정책 ======
RISK_LEVEL = { "Human": "high", "Wild Boar": "high", "Dog": "high", "Water Deer": "high", "Racoon": "high", "Leopard Cat": "high", "Squirrel": "high", "Weasel": "high", "Wild Rabbit": "high", "Bird": "high" }
RISK_POLICY = { "high": {"persist": 2.0}, "medium": {"persist": 2.5}, "low": {"persist": 3.0} }
def risk_level_of(name_norm: str) -> str: return RISK_LEVEL.get(name_norm, "low")
def persist_sec_of(name_norm: str) -> float: return RISK_POLICY[risk_level_of(name_norm)]["persist"]

# ▼▼▼ 키를 전부 normalize 형태로 통일 ▼▼▼
RISK_LEVEL    = { normalize(k): v for k, v in RISK_LEVEL.items() }
CLASS_CONF_TH = { normalize(k): v for k, v in CLASS_CONF_TH.items() }

MONITOR_CLASSES = None   # e.g., {"human", "dog", "wild boar"}

# =========== Camera / Display / Model ===========
camera = jetson_utils.gstCamera(WIDTH, HEIGHT, "/dev/video0")
try:
    display_local = jetson_utils.videoOutput()
except Exception:
    display_local = None

display_rtmp  = jetson_utils.videoOutput("rtmp://agrilook-be-stream.koreacentral.cloudapp.azure.com:1935/live/jetson3")

net = jetson_inference.detectNet(argv=["--model=/home/jetson/vigil/mb2-ssd-lite_512_full.onnx", f"--labels={LABELS_PATH}", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes", "--threshold=0.25", "--verbose"])
tracker = BYTETrackerKF(track_thresh=0.30,        # ↓ 매칭에 쓰일 high det 더 포함
                        match_thresh=0.25,        # ↓ IoU 매칭 문턱 조금 더 느슨
                        match_thresh_low=0.15,    # ↓ 저신뢰 fallback도 더 관대
                        buffer_ttl=300,           # ↑ 트랙을 더 오래 유지
                        min_box_area=DEFAULT_MIN_AREA,
                        gate_maha=True,           # ↑ 칼만 예측 중심으로 게이팅 사용
                        maha_th=100.0,            # ↑ 큰 움직임도 허용(게이트 넓힘)
                        dt=1.0,
                        class_consistent=True)    # ↑ 클래스가 흔들려도 섞이지 않게

font = jetson_utils.cudaFont()

# === PoseNet 초기화 ===
pose_net = None
if ENABLE_POSENET:
    try:
        pose_net = jetson_inference.poseNet("resnet18-body", argv=["--logging=0"])
        print("[POSE] poseNet ready: resnet18-body")
    except Exception as e:
        print("[POSE] init failed:", e)
        pose_net = None


# === SCRIPT 1에서 병합된 캡처 및 오디오 상태 관리 START ===
CAPTURE_DIR = "/home/jetson/vigil/captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# (A) 실제 파일 경로: "사운드 키 → mp3 파일"
SOUND_FILES = {
    "voice": os.path.join(BASE_DIR, "voice.mp3"),
    "alarm": os.path.join(BASE_DIR, "alarm.mp3"),
    "dog":   os.path.join(BASE_DIR, "dog.mp3"),
    "wolf":  os.path.join(BASE_DIR, "wolf.mp3"),
    "tiger": os.path.join(BASE_DIR, "tiger.mp3")
}
# 부팅/시작 시 모든 MP3를 증폭본으로 교체
for key, path in list(SOUND_FILES.items()):
    desired = SOUND_GAIN_DB.get(key, 0)
    if os.path.exists(path):
        applied = safe_boost_db(path, desired)
        if applied > 0:
            SOUND_FILES[key] = make_boosted_copy(path, applied)
            print(f"[AUDIO] using boosted '{key}': +{applied} dB -> {SOUND_FILES[key]}")
        else:
            print(f"[AUDIO] using original '{key}' (no headroom or no boost)")


# (B) 클래스별 재생 맵핑: "탐지 클래스(정규화/스네이크) → 사운드 키"
#  * sanitize_label("Siberian Chipmunk") = "siberian_chipmunk"
#  * sanitize_label("Racoon") = "racoon"  (라쿤 스펠링이 코드에서 'Racoon'임에 주의)
SOUND_MAP = {
    "human":               "voice",   # human  → voice.mp3

    "dog":                 "alarm",   # dog    → alarm.mp3

    "bird":                "dog",     # bird   → dog.mp3

    "squirrel":            "wolf",    # 아래 3종 → wolf.mp3
    "racoon":              "wolf",
    "siberian_chipmunk":   "wolf",

    "wild_boar":           "tiger",   # 아래 5종 → tiger.mp3
    "leopard_cat":         "tiger",
    "water_deer":          "tiger",
    "weasel":              "tiger",
    "wild_rabbit":         "tiger"
}

AUDIO_COOLDOWN_SEC = 5.0
last_audio_ts_sound = {}   # {사운드키: 마지막 재생시각}
audio_playing  = False
audio_started  = 0.0
audio_len_sec  = 0.0
# === SCRIPT 1에서 병합된 캡처 및 오디오 상태 관리 END ===

seen_since = {}

ok_last_seen = {}  # {track_id: last_ok_timestamp}
ok_tracks = set()  # remember OK trackID

def is_ok_active(tid: int, now_ts: float) -> bool:
    """OK 제스처가 최근 OK_MEMORY_HANG_SEC 내에 관측되었는지"""
    last = ok_last_seen.get(tid, 0.0)
    return (tid in ok_tracks) and (now_ts - last <= OK_MEMORY_HANG_SEC)

captured_ids = set()
start = time.time()
frame_idx = 0

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

    try:
        kind, det_class = prefix.split("_", 1)
    except ValueError:
        kind, det_class = prefix, ""
    filename = f"{kind}_{dt_str}{'_' + det_class if det_class else ''}.jpg"

    local_path = os.path.join(persist_dir, filename)
    jetson_utils.saveImage(local_path, crop_gpu)
    try:
        blob_url = upload_file_to(local_path, filename, container=CONT_CROP, sas_url=SAS_CROP)
        print(f"[BLOB] uploaded: {blob_url}")
    except Exception as e: print(f"[BLOB] upload failed: {e}"); blob_url = ""
    return local_path, blob_url
# =================================================================
def full_save_and_upload(img, dt_str, prefix="full", persist_dir=CAPTURE_DIR):
    os.makedirs(persist_dir, exist_ok=True)

    try:
        kind, det_class = prefix.split("_", 1)
    except ValueError:
        kind, det_class = prefix, ""
    filename = f"{kind}_{dt_str}{'_' + det_class if det_class else ''}.jpg"

    local_path = os.path.join(persist_dir, filename)
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

        # 4) 렌더링 / 캡처 / DB 저장 / 경고음 재생
        now = time.time()
        # === OK 만료 정리 ===
        for _tid in list(ok_tracks):
            if not is_ok_active(_tid, now):
                ok_tracks.discard(_tid)

        frame_idx += 1              # ★ 추가: 프레임 카운터 증가 (사람 N프레임마다 PoseNet용)
        active_ids = set()
        for t in tracks:
            tid = t.id; active_ids.add(tid)
            track_last_seen[tid] = now      # 추가: 방금 이 트랙을 봤다고 기록
            x1, y1, x2, y2 = t.tlbr
            if tid not in seen_since: seen_since[tid] = now
            elapsed = now - seen_since[tid]
            cname = id_to_label[t.class_id] if 0 <= t.class_id < len(id_to_label) else str(t.class_id)
            cname_norm = normalize(cname)

            # ── 기존 박스/라벨 렌더 ──
            draw_box(img, x1, y1, x2, y2, id2color(tid))
            label_bg = class2color(t.class_id if 0 <= t.class_id < len(id_to_label) else 0)
            label_text = f"{cname} | id:{tid} | s:{t.score:.2f} | t:{elapsed:.1f}"
            tx, ty = int(max(0, x1)), int(y1 - 22) if y1 >= 22 else int(y1 + 2)
            draw_label_with_bg(img, w, h, tx, ty, label_text, bg_color=label_bg)

            # === PoseNet in human bbox (face hidden, limbs only, + OK gesture) ===
            if ENABLE_POSENET and pose_net is not None and HUMAN_ID is not None:
                if t.class_id == HUMAN_ID and (frame_idx % POSENET_PER_HUMAN_EVERY_N_FRAMES == 0):
                    try:
                        pad = int(POSE_PADDING)
                        bx1 = max(0, int(x1) - pad)
                        by1 = max(0, int(y1) - pad)
                        bx2 = min(int(x2) + pad, int(w) - 1)
                        by2 = min(int(y2) + pad, int(h) - 1)

                        cw = max(1, bx2 - bx1)
                        ch = max(1, by2 - by1)
                        if cw < 60 or ch < 60:
                            # 너무 작으면 PoseNet 스킵
                            pass
                        else:
                            fmt = getattr(img, 'format', 'rgba32f')
                            crop_gpu = jetson_utils.cudaAllocMapped(width=cw, height=ch, format=fmt)
                            jetson_utils.cudaCrop(img, crop_gpu, (bx1, by1, bx2, by2))

                            # overlay=none 으로 포즈만 받아서 '원본 프레임'에 직접 그린다
                            poses = pose_net.Process(crop_gpu, overlay="none")

                            # 한 사람만 처리 (가장 신뢰도 높은 하나 선택)
                            target_pose = None
                            if poses:
                                target_pose = max(poses, key=lambda p: getattr(p, "Confidence", 1.0))

                            if target_pose is not None:
                                # crop 좌표 → 원본 좌표 보정을 위한 헬퍼
                                def _kp_xy(kp):
                                    return int(bx1 + kp.x), int(by1 + kp.y), float(getattr(kp, "Confidence", 1.0))

                                # 키포인트 맵: {id: (x,y,conf)}
                                kp_map = {}
                                for kp in target_pose.Keypoints:
                                    xk, yk, ck = _kp_xy(kp)
                                    kp_map[int(kp.ID)] = (xk, yk, ck)

                                # (1) 얼굴 키포인트는 "점"도 그리지 않음
                                for kid, (kx, ky, kc) in kp_map.items():
                                    if kid in FACE_IDS or kc < POSE_KP_TH:
                                        continue
                                    jetson_utils.cudaDrawCircle(img, (kx, ky), 3, (0, 255, 255, 255))  # 노란 점

                                # (2) 얼굴과 연결된 링크는 그리지 않고, LIMB_EDGES만 그린다
                                for a, b in LIMB_EDGES:
                                    if a in kp_map and b in kp_map:
                                        (ax, ay, ac) = kp_map[a]
                                        (bxp, byp, bc) = kp_map[b]
                                        if ac >= POSE_KP_TH and bc >= POSE_KP_TH:
                                            jetson_utils.cudaDrawLine(img, (ax, ay), (bxp, byp), (0, 255, 0, 255))  # 초록 라인

                                # (3) OK 제스처 감지 (양손 머리 위에서 서로 가깝게 원형)
                                # 필요 키포인트
                                need = [5, 6, 7, 8, 9, 10, 0]  # 양어깨/팔/손목/코(머리 높이 대용)
                                if all(k in kp_map for k in need):
                                    (lsx, lsy, lsc) = kp_map[5]   # left_shoulder
                                    (rsx, rsy, rsc) = kp_map[6]   # right_shoulder
                                    (lwx, lwy, lwc) = kp_map[9]   # left_wrist
                                    (rwx, rwy, rwc) = kp_map[10]  # right_wrist
                                    (lox, loy, loc) = kp_map[7]   # left_elbow
                                    (rox, roy, roc) = kp_map[8]   # right_elbow
                                    (nx, ny, nc)    = kp_map.get(0, (0, 0, 0.0))  # nose (optional)

                                    if min(lsc, rsc, lwc, rwc, loc, roc) >= POSE_KP_TH:
                                        # 스케일: 어깨 폭
                                        shoulder_w = max(1.0, ((lsx - rsx)**2 + (lsy - rsy)**2) ** 0.5)

                                        # 조건: 양 손목이 어깨보다 충분히 위 (y가 작음), 서로도 충분히 가까움
                                        wrists_above_shoulder = (lwy < min(lsy, rsy) - 0.15 * shoulder_w) and \
                                                                (rwy < min(lsy, rsy) - 0.15 * shoulder_w)
                                        wrists_close = (((lwx - rwx)**2 + (lwy - rwy)**2) ** 0.5) < (1.2 * shoulder_w)
                                        elbows_above_shoulder = (loy < min(lsy, rsy) - 0.05 * shoulder_w) and \
                                                                (roy < min(lsy, rsy) - 0.05 * shoulder_w)

                                        is_ok = wrists_above_shoulder and wrists_close and elbows_above_shoulder

                                        if is_ok:
                                            ok_last_seen[tid] = time.time()
                                            ok_tracks.add(tid)

                                # (4) OK 스티키 표시
                                if tid in ok_tracks:
                                    # bbox 좌상단에 초록 라벨로 크게
                                    draw_label_with_bg(img, w, h, int(x1), int(max(0, y1 - 44)), "OK", bg_color=(0, 180, 0, 200))

                    except Exception as e:
                        print("[POSE] bbox-crop posenet failed:", e)

            # 위험도 기반 지속 시간 체크 후 캡처/업로드/DB저장
            need_sec = persist_sec_of(cname_norm)

            # 사람 + OK 활성 => 캡쳐/DB 차단
            ok_silence = (cname_norm == "human") and is_ok_active(tid, now)
            if ok_silence:
                # 시각화(선택): 'OK (silenced)' 라벨
                draw_label_with_bg(img, w, h, int(x1), int(max(0, y1 - 44)),
                                   "OK (silenced)", bg_color=(0, 180, 0, 220))

            # 캡쳐/DB는 not ok_silence일 때만 실행
            if (not ok_silence) and (elapsed >= need_sec) and (tid not in captured_ids):
                try:
                    label   = sanitize_label(cname)
                    dt_str  = datetime.now(KST).strftime("%Y%m%d-%H%M%S")
                    conf_pct = int(round(t.score * 100))
                    conf_str = f"{conf_pct}%"

                    # 1) 이미지 저장/업로드는 항상 시도
                    crop_local, crop_url = crop_save_and_upload(
                        img, dt_str, (x1, y1, x2, y2),
                        prefix=label_prefix(label, crop=True)
                    )
                    full_local, full_url = full_save_and_upload(
                        img, dt_str,
                        prefix=label_prefix(label, crop=False)
                    )

                    # 2) DB 저장은 "연결되어 있을 때만" 실행, 아니면 이유를 로그로 출력
                    if db_ok and col is not None:
                        try:
                            seq_str = next_id_in_same_collection(width=4)
                            col.insert_one({
                                "_id": seq_str,
                                "class": label,
                                "confidence": conf_str,
                                "datetime": dt_str,
                                "farm_id": FARM_ID,
                            })
                            print(f"[DB] inserted _id={seq_str} class={label} conf={conf_str} datetime={dt_str}")
                        except DuplicateKeyError:
                            pass
                        except Exception as e:
                            print(f"[DB] insert failed -> skip next DB writes this run: {e}")
                            db_ok = False  # 이후부터 DB 시도 중지
                    else:
                        now_ts = time.time()
                        # 스팸 방지: 최근 {DB_SKIP_LOG_THROTTLE}초 이내면 같은 경고 생략
                        if now_ts - last_db_skip_log >= DB_SKIP_LOG_THROTTLE:
                            print(f"[DB] SKIP (disconnected): class={label} conf={conf_str} datetime={dt_str}")
                            last_db_skip_log = now_ts

                    captured_ids.add(tid)

                except Exception as e:
                    # 어떤 이유로든 이 블록이 실패해도 앱이 죽지 않도록 방어
                    print(f"[CAPTURE] unexpected error (ignored): {e}")

        for tid in list(seen_since.keys()):
            if tid not in active_ids:
                del seen_since[tid]

        # === SCRIPT 1에서 병합된 클래스별 오디오 재생 로직 START ===
        ready_classes_by_sound = defaultdict(list)

        for t in tracks:
            if t.id not in seen_since:
                continue

            # 사람 + OK 활성 => 경고음 후보 제외
            if 0 <= t.class_id < len(id_to_label):
                cname_h = id_to_label[t.class_id]
                if normalize(cname_h) == "human" and is_ok_active(t.id, now):
                    # print(f"[OK] human id={t.id} silenced -> skip sound")  # (선택) 로그
                    continue

            cname = id_to_label[t.class_id] if 0 <= t.class_id < len(id_to_label) else str(t.class_id)
            cname_norm_policy = normalize(cname)       # 정책용(스페이스, 소문자)
            cname_norm_audio  = sanitize_label(cname)  # 오디오 매핑용(스네이크)

            need_sec = persist_sec_of(cname_norm_policy)
            elapsed  = now - seen_since[t.id]

            # 재생 조건 충족 시, 이 클래스가 트리거하는 사운드 키를 누적
            if elapsed >= need_sec:
                sound_key = SOUND_MAP.get(cname_norm_audio)
                if sound_key and sound_key in SOUND_FILES and os.path.exists(SOUND_FILES[sound_key]):
                    ready_classes_by_sound[sound_key].append(cname_norm_audio)

        # 준비된 사운드 중에서 우선순위에 따라 하나 선택
        play_order   = SOUND_PRIORITY if 'SOUND_PRIORITY' in globals() else list(SOUND_FILES.keys())
        chosen_sound = next((key for key in play_order if key in ready_classes_by_sound), None)

        if chosen_sound:
            last_ts = last_audio_ts_sound.get(chosen_sound, 0.0)
            if (not audio_playing) and (now - last_ts) >= AUDIO_COOLDOWN_SEC:
                mp3_file = SOUND_FILES[chosen_sound]
                # ▶▶ 여기서 “어떤 클래스 때문에 어떤 mp3가 재생되는지”를 로그로 출력
                classes_str = ", ".join(sorted(set(ready_classes_by_sound[chosen_sound])))

                audio_len_sec = play_mp3_once(mp3_file, volume=100)
                print(f"[AUDIO] using file: {mp3_file}")
                print(f"[AUDIO] {classes_str} detected -> {os.path.basename(mp3_file)} ({audio_len_sec:.2f}s) playing....")
                audio_started = time.time()
                audio_playing = True
                last_audio_ts_sound[chosen_sound] = now

        # VLC 상태로 종료 감지 (타이머로 자르지 않음)
        try:
            state = player.get_state()
        except Exception:
            state = None

        # 재생이 정상 종료/중지/에러 상태이거나, 시작 후 조금 지났는데 is_playing()==0 이면 끝난 것으로 처리
        if audio_playing and (
            (state in (vlc.State.Ended, vlc.State.Stopped, vlc.State.Error))
            or (time.time() - audio_started > 0.5 and player.is_playing() == 0)
        ):
            audio_playing = False

        # === SCRIPT 1에서 병합된 클래스별 오디오 재생 로직 END ===

        # 5) 출력
        if display_local and display_local.IsStreaming():
            display_local.Render(img)
            display_local.SetStatus(f"{net.GetNetworkFPS():.0f} FPS | det_raw={len(dets)} | det_used={len(det_list)} | tracks={len(tracks)}")
        if display_rtmp:
            display_rtmp.Render(img)

        if time.time() - start > 3600: # Timeout 설정 (필요시 조정)
            print("[TIMEOUT] Timeout exit ===========================")
            break

except KeyboardInterrupt: print("Interrupted")
finally:
    # 모든 리소스 정리
    for resource in [camera, display_rtmp, display_local]: # display_local
        if resource is not None:
            try: resource.Close()
            except Exception: pass
    try: player.stop()
    except Exception: pass
    print("Done.")
