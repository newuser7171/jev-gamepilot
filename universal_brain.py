"""
Universal Jev / Laya System One Brain for Jev-GamePilot.
Combines:
1. High-speed local Laya decision engine (convaiinnovations/laya) for sub-30ms offline inference.
2. Local openjev (AlexWortega/openjev Qwen3.5 NLI cross-encoder) typed-decision arbitration (fast path).
3. Local LLM2Jev Transformers backend (LLM2JEV_MODEL) for deep offline Choice/Score/Noul arbitration when openjev is weak.
4. Local GLiNER2.5-Decide (fastino 340M label-set classifier) as mid-tier when openjev is weak.
5. Local Bev (Reza2kn/Bev Ternary-Bonsai-2-27B) SystemOne HTTP decision API on loopback.
6. TypeSafe Jev System One SDK cloud API (jev-latest) as robust online fallback.
7. Simple Jev Featherless classifier API (Choice/Score/Noul; keyless demo or FEATHERLESS_API_KEY).
8. Keyless classifier.dev fallback for zero-config environments (circuit-broken on hard HTTP failures).
7. Intelligent spatial & aerodynamic reflex actuator (no blind defaults; altitude & lane aware).
8. Continuous async worker for 60+ FPS lock-free gameplay.
"""

import json
import math
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from profile_manager import GameAction, GameProfile
from universal_vision import UniversalEntity, UniversalSceneState

load_dotenv()


def _env_float(name: str, default: float) -> float:
    """Parse a float env override; fall back to default on blank/invalid."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Passive actions never commit on a high-threat fast path.
_PASSIVE_ACTIONS = frozenset({"wait", "maintain_course", "stand_idle"})

# Try importing local laya engine (supports both Apple Silicon MLX and standard PyTorch)
_LAYA_AVAILABLE = False
_LAYA_BACKEND = "none"
try:
    import laya_mlx as laya
    _LAYA_AVAILABLE = True
    _LAYA_BACKEND = "mlx"
except ImportError:
    try:
        import laya
        _LAYA_AVAILABLE = True
        _LAYA_BACKEND = "pytorch"
    except ImportError:
        pass

# Try importing TypeSafe SDK
_TYPESAFE_AVAILABLE = False
try:
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
    _TYPESAFE_AVAILABLE = True
except ImportError:
    pass

# Local LLM2Jev Transformers backend (CPU, LLM2JEV_MODEL env path / HF id).
# Aliased to avoid colliding with typesafe_sdk Choice/Score/Noul.
_LLM2JEV_AVAILABLE = False
try:
    from llm2jev import (
        Choice as L2JChoice,
        JevRequest as L2JJevRequest,
        LLM2Jev as L2JEngine,
        Noul as L2JNoul,
        Score as L2JScore,
        TransformersBackend as L2JTransformersBackend,
    )
    _LLM2JEV_AVAILABLE = True
except ImportError:
    L2JChoice = L2JScore = L2JNoul = None
    L2JJevRequest = L2JEngine = L2JTransformersBackend = None

# openjev — AlexWortega/openjev Qwen3.5 NLI cross-encoder as a local typed-decision backend.
_OPENJEV_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "openjev")
_OPENJEV_SUBFOLDER = "qwen3.5-0.8b-nli-v5"
_OPENJEV_AVAILABLE = False
try:
    if os.path.isdir(os.path.join(_OPENJEV_ROOT, _OPENJEV_SUBFOLDER)):
        if _OPENJEV_ROOT not in sys.path:
            sys.path.insert(0, _OPENJEV_ROOT)
        from openjev_decide import OpenJev as OpenJevEngine  # noqa: E402

        _OPENJEV_AVAILABLE = True
except Exception:
    OpenJevEngine = None

# GLiNER2.5-Decide — fastino local label-set classifier (340M DeBERTa, gliner2 API).
_GLINER_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "gliner25_decide")
_GLINER_AVAILABLE = False
try:
    from gliner2.classification import ClassificationSchema, Classifier

    _GLINER_AVAILABLE = True
except Exception:
    ClassificationSchema = Classifier = None

# Try importing ClashBattleAdapter
_CLASH_ADAPTER_AVAILABLE = False
try:
    from adapters.clash_adapter import ClashBattleAdapter
    _CLASH_ADAPTER_AVAILABLE = True
except Exception:
    pass

_COC_ADAPTER_AVAILABLE = False
try:
    from adapters.coc_adapter import CocRaidAdapter, detect_coc_phase
    _COC_ADAPTER_AVAILABLE = True
except Exception:
    detect_coc_phase = None

_BRAWL_ADAPTER_AVAILABLE = False
try:
    from adapters.brawl_adapter import BrawlMatchAdapter, detect_brawl_phase
    _BRAWL_ADAPTER_AVAILABLE = True
except Exception:
    detect_brawl_phase = None


class UniversalBrain:
    def __init__(self, model_name: str = "jev-latest"):
        self.model_name = model_name
        # The SDK is optional: installed does not mean credentials are configured.
        # Leave the client unset so the existing keyless/local fallbacks can run.
        self.client = (
            TypeSafeClient()
            if _TYPESAFE_AVAILABLE and os.getenv("TYPESAFE_API_KEY", "").strip()
            else None
        )
        self.laya_agent = None
        self.laya_loading = False
        self._laya_tried = False
        # Fusion tunables (env-overridable without code edits).
        self.laya_fast_path_conf = _env_float("LAYA_FAST_PATH_CONF", 0.88)
        # openjev alone can skip the 50s+ LLM2Jev path when it is this sure.
        self.local_accept_conf = _env_float("LOCAL_ACCEPT_CONF", 0.72)
        # Threat level that blocks a passive Laya fast-path commit.
        self.fast_path_threat_gate = _env_float("FAST_PATH_THREAT_GATE", 0.60)
        self.llm2jev_model = os.getenv("LLM2JEV_MODEL", "").strip()
        self.llm2jev_engine = None
        self.llm2jev_loading = False
        self._llm2jev_tried = False
        self._llm2jev_lock = threading.Lock()
        # openjev local NLI cross-encoder (typed decisions over profile.actions).
        self.openjev_engine = None
        self.openjev_loading = False
        self._openjev_tried = False
        self._openjev_lock = threading.Lock()
        self.openjev_path = os.getenv("OPENJEV_PATH", _OPENJEV_ROOT).strip() or _OPENJEV_ROOT
        self.openjev_subfolder = os.getenv("OPENJEV_SUBFOLDER", _OPENJEV_SUBFOLDER).strip() or _OPENJEV_SUBFOLDER
        # GLiNER2.5-Decide — lazy mid-tier label-set classifier (loads on first weak openjev).
        self.gliner_engine = None
        self.gliner_loading = False
        self._gliner_tried = False
        self._gliner_lock = threading.Lock()
        self.gliner_path = os.getenv("GLINER_PATH", _GLINER_ROOT).strip() or _GLINER_ROOT
        # Simple Jev — Featherless classifier cloud tier (keyless demo, or FEATHERLESS_API_KEY).
        self.simple_jev_url = (
            os.getenv("SIMPLE_JEV_URL", "https://simple-jev-demo-api.featherless.ai").strip().rstrip("/")
            or "https://simple-jev-demo-api.featherless.ai"
        )
        self.simple_jev_model = (
            os.getenv("SIMPLE_JEV_MODEL", "featherless-ai/Qwen3.5-4B-classifier").strip()
            or "featherless-ai/Qwen3.5-4B-classifier"
        )
        self.simple_jev_key = os.getenv("FEATHERLESS_API_KEY", "").strip()
        self._simple_jev_dead = False
        self._simple_jev_last_at = 0.0
        # Bev — Reza2kn/Bev decision service (Ternary-Bonsai-2-27B) on loopback HTTP.
        self.bev_url = os.getenv("BEV_URL", "http://127.0.0.1:18781").rstrip("/") or "http://127.0.0.1:18781"
        self.bev_model = os.getenv("BEV_MODEL", "bev-bonsai-27b").strip() or "bev-bonsai-27b"
        self._bev_lock = threading.Lock()
        self._bev_up = False
        self._bev_checked_at = 0.0
        self.clash_adapter = ClashBattleAdapter() if _CLASH_ADAPTER_AVAILABLE else None
        self.coc_adapter = CocRaidAdapter() if _COC_ADAPTER_AVAILABLE else None
        self.brawl_adapter = BrawlMatchAdapter() if _BRAWL_ADAPTER_AVAILABLE else None

        self.last_decision: Dict[str, Any] = {
            "action": "wait",
            "threat_score": 0.0,
            "confidence": 1.0,
            "latency_ms": 0.0,
            "source": "init",
            "target_coords": None,
        }

        self._lock = threading.Lock()
        self._work_queue: queue.Queue = queue.Queue(maxsize=2)
        self._classifier_dead = False
        self._worker_thread = threading.Thread(
            target=self._continuous_worker, daemon=True, name="UniversalBrainWorker"
        )
        self._worker_thread.start()

        # Attempt background local Laya load
        self._start_laya_loader()
        # Background LLM2Jev load only when LLM2JEV_MODEL is set (opt-in).
        self._start_llm2jev_loader()
        # Background openjev load (local weights under models/openjev).
        self._start_openjev_loader()

    def _start_llm2jev_loader(self):
        """Background-loads LLM2Jev Transformers backend when LLM2JEV_MODEL is set."""
        if not _LLM2JEV_AVAILABLE or not self.llm2jev_model or self._llm2jev_tried:
            return

        def loader():
            self._llm2jev_tried = True
            self.llm2jev_loading = True
            try:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                os.environ["HF_HUB_DISABLE_XET"] = "1"
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                path = self.llm2jev_model
                if not os.path.isdir(path):
                    from huggingface_hub import snapshot_download

                    path = snapshot_download(path)
                backend = L2JTransformersBackend(path, batch_size=16)
                engine = L2JEngine(backend=backend)
                with self._lock:
                    self.llm2jev_engine = engine
                    self.llm2jev_model = path
            except Exception:
                with self._lock:
                    self.llm2jev_engine = None
            finally:
                self.llm2jev_loading = False

        threading.Thread(target=loader, daemon=True, name="LLM2JevLoader").start()

    def _start_openjev_loader(self):
        """Background-load AlexWortega/openjev Qwen3.5 NLI cross-encoder (local weights)."""
        if not _OPENJEV_AVAILABLE or self._openjev_tried:
            return
        self._openjev_tried = True
        self.openjev_loading = True

        def loader():
            try:
                engine = OpenJevEngine.from_pretrained(
                    self.openjev_path, subfolder=self.openjev_subfolder, device="cpu"
                )
                with self._lock:
                    self.openjev_engine = engine
            except Exception:
                with self._lock:
                    self.openjev_engine = None
            finally:
                self.openjev_loading = False

        threading.Thread(target=loader, daemon=True, name="OpenJevLoader").start()

    def _start_gliner_loader(self):
        """Background-load fastino/GLiNER2.5-Decide classifier (lazy; first weak openjev)."""
        if not _GLINER_AVAILABLE or self._gliner_tried:
            return
        self._gliner_tried = True
        self.gliner_loading = True

        def loader():
            try:
                # gliner2 prints a Unicode banner; Windows cp1252 stdout would crash it.
                import contextlib
                import io

                with contextlib.redirect_stdout(io.StringIO()):
                    if os.path.isdir(self.gliner_path):
                        engine = Classifier.from_pretrained(self.gliner_path)
                    else:
                        engine = Classifier.from_pretrained("fastino/GLiNER2.5-Decide")
                with self._lock:
                    self.gliner_engine = engine
            except Exception:
                with self._lock:
                    self.gliner_engine = None
            finally:
                self.gliner_loading = False

        threading.Thread(target=loader, daemon=True, name="GLiNERLoader").start()

    def _start_laya_loader(self):
        """Asynchronously attempts to load local Laya model weights without network downloads."""
        if not _LAYA_AVAILABLE or self.laya_loading or self._laya_tried:
            return

        def loader():
            self.laya_loading = True
            self._laya_tried = True
            # Suppress all HF download progress bars and warnings
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

            # 1. Load directly from local offline cached snapshot
            local_dir = self._find_local_laya_checkpoint()
            if local_dir:
                try:
                    agent = laya.load(local_dir)
                    with self._lock:
                        self.laya_agent = agent
                    return
                except Exception:
                    pass

            self.laya_loading = False

        threading.Thread(target=loader, daemon=True, name="LayaModelLoader").start()

    @staticmethod
    def _find_local_laya_checkpoint() -> Optional[str]:
        """Locates the local offline Laya checkpoint in HuggingFace cache or MLX model directories."""
        # 1. MLX directories (Apple Silicon)
        if _LAYA_BACKEND == "mlx":
            mlx_dirs = [
                "models/hub/laya-multilingual-mlx",
                "models/laya-multilingual",
                os.path.expanduser(r"~\.cache\huggingface\hub\models--aac6fef--laya-multilingual-mlx"),
                os.path.expanduser(r"~\.cache\huggingface\hub\models--aac6fef--laya-mlx"),
                "aac6fef/laya-multilingual-mlx",
            ]
            for d in mlx_dirs:
                if os.path.exists(d):
                    return d
            return "aac6fef/laya-multilingual-mlx"

        # 2. PyTorch HuggingFace cache
        hf_base = os.path.expanduser(r"~\.cache\huggingface\hub\models--convaiinnovations--laya\snapshots")
        if os.path.isdir(hf_base):
            for snap in os.listdir(hf_base):
                snap_path = os.path.join(hf_base, snap)
                multi = os.path.join(snap_path, "multilingual")
                if os.path.exists(os.path.join(multi, "model.safetensors")):
                    return multi
                if os.path.exists(os.path.join(snap_path, "model.safetensors")):
                    return snap_path
        return None

    def _continuous_worker(self):
        """Continuous background worker processing scene evaluations without thread churn."""
        while True:
            try:
                task = self._work_queue.get()
                if task is None:
                    break
                profile, scene = task
                self._evaluate_scene(profile, scene)
            except Exception as e:
                time.sleep(0.02)

    def _evaluate_scene(self, profile: GameProfile, scene: UniversalSceneState):
        """
        Unified Laya + Jev Dual-System Architecture:
        1. Fast local in-process inference via Laya (15-25ms, zero network overhead).
        2. If Laya confidence is high (>= LAYA_FAST_PATH_CONF, default 0.88) AND
           not a passive commit under live threat, commit immediately.
        3. Escalation: local openjev first (fast NLI cross-encoder), then local
           GLiNER2.5-Decide (label-set classifier), then local LLM2Jev only when
           both are missing/weak (50s+ CPU), then Bev, TypeSafe cloud, Simple Jev,
           then classifier.dev.
        4. If both provide decisions, fuse them into 'laya_jev_consensus' with blended confidence.
        """
        laya_res = None
        if self.laya_agent is not None:
            laya_res = self._query_local_laya(profile, scene)

        # Fast path: high-confidence Laya — but never hard-commit a passive
        # action when the scene or Laya itself reports live threat.
        if laya_res and self._laya_fast_path_ok(laya_res, scene):
            with self._lock:
                self.last_decision = laya_res
            return

        # Secondary opinion / Escalation: local openjev → GLiNER → LLM2Jev, then local Bev
        # (loopback SystemOne), then TypeSafe cloud, Simple Jev, then classifier.dev.
        jev_res = self._local_escalation(profile, scene)
        if not jev_res:
            jev_res = self._query_bev(profile, scene)
        if not jev_res and self.client is not None and os.getenv("TYPESAFE_API_KEY"):
            jev_res = self._query_typesafe_sdk(profile, scene)
        if not jev_res and not self._simple_jev_dead:
            jev_res = self._query_simple_jev(profile, scene)
        if not jev_res and not self._classifier_dead:
            jev_res = self._query_classifier_dev(profile, scene)

        # Consensus Fusion: Merge Laya + Jev decisions
        if laya_res and jev_res:
            laya_act = laya_res.get("action")
            jev_act = jev_res.get("action")
            if laya_act == jev_act:
                # Both Laya and Jev agree: maximum confidence consensus
                fused = {
                    "action": laya_act,
                    "threat_score": max(laya_res.get("threat_score", 0), jev_res.get("threat_score", 0)),
                    "confidence": min(0.99, round((laya_res.get("confidence", 0.8) + jev_res.get("confidence", 0.8)) / 2 + 0.1, 2)),
                    "latency_ms": round(laya_res.get("latency_ms", 15) + jev_res.get("latency_ms", 50), 1),
                    "source": "laya_jev_consensus",
                    "target_coords": laya_res.get("target_coords") or jev_res.get("target_coords"),
                }
            else:
                fused = self._resolve_disagreement(laya_res, jev_res, scene)

            with self._lock:
                self.last_decision = fused
            return

        # Fallback to whichever responded
        winner = laya_res or jev_res
        if winner:
            with self._lock:
                self.last_decision = winner

    def _laya_fast_path_ok(self, laya_res: Dict[str, Any], scene: UniversalSceneState) -> bool:
        """True when Laya may commit without escalation.

        Blocks passive actions under live threat so a high-confidence
        'wait' cannot freeze the pilot while units close in.
        """
        conf = float(laya_res.get("confidence", 0.0) or 0.0)
        if conf < self.laya_fast_path_conf:
            return False
        if laya_res.get("action") not in _PASSIVE_ACTIONS:
            return True
        scene_urgency = float(getattr(scene, "threat_urgency", 0.0) or 0.0)
        laya_threat = float(laya_res.get("threat_score", 0.0) or 0.0)
        return scene_urgency < self.fast_path_threat_gate and laya_threat < self.fast_path_threat_gate

    def _local_escalation(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """openjev first (fast); GLiNER mid-tier on first weak openjev; LLM2Jev last."""
        fast = None
        if self.openjev_engine is not None:
            fast = self._query_local_openjev(profile, scene)
        if fast and float(fast.get("confidence", 0.0) or 0.0) >= self.local_accept_conf:
            return fast
        # Lazy-load GLiNER mid tier only when openjev is missing or unsure.
        if self.gliner_engine is None and not self._gliner_tried:
            self._start_gliner_loader()
        mid = None
        if self.gliner_engine is not None:
            mid = self._query_gliner_decide(profile, scene)
            if mid and float(mid.get("confidence", 0.0) or 0.0) >= self.local_accept_conf:
                return mid
        if self.llm2jev_engine is None:
            return fast or mid
        deep = self._query_local_llm2jev(profile, scene)
        if not deep:
            return fast or mid
        best = fast or mid
        if not best or float(deep.get("confidence", 0) or 0) > float(best.get("confidence", 0) or 0):
            return deep
        return best

    def _resolve_disagreement(
        self,
        laya_res: Dict[str, Any],
        jev_res: Dict[str, Any],
        scene: UniversalSceneState,
    ) -> Dict[str, Any]:
        """Pick between disagreeing opinions; under high threat prefer non-passive."""
        laya_c = float(laya_res.get("confidence", 0) or 0)
        jev_c = float(jev_res.get("confidence", 0) or 0)
        scene_urgency = float(getattr(scene, "threat_urgency", 0.0) or 0.0)
        if scene_urgency >= self.fast_path_threat_gate:
            laya_passive = laya_res.get("action") in _PASSIVE_ACTIONS
            jev_passive = jev_res.get("action") in _PASSIVE_ACTIONS
            if laya_passive and not jev_passive and jev_c >= laya_c - 0.15:
                fused = dict(jev_res)
                fused["source"] = "threat_preferred_jev"
                return fused
            if jev_passive and not laya_passive and laya_c >= jev_c - 0.15:
                fused = dict(laya_res)
                fused["source"] = "threat_preferred_laya"
                return fused
        if laya_c >= jev_c:
            fused = dict(laya_res)
            fused["source"] = "laya_preferred"
        else:
            fused = dict(jev_res)
            fused["source"] = "jev_escalation"
        return fused

    def query_jev_universal(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """Synchronously queries combined Laya + Jev fusion for testing / benchmarking."""
        laya_res = None
        if self.laya_agent is not None:
            laya_res = self._query_local_laya(profile, scene)

        if laya_res and self._laya_fast_path_ok(laya_res, scene):
            with self._lock:
                self.last_decision = laya_res
            return laya_res

        jev_res = self._local_escalation(profile, scene)
        if not jev_res:
            jev_res = self._query_bev(profile, scene)
        if not jev_res and self.client is not None and os.getenv("TYPESAFE_API_KEY"):
            jev_res = self._query_typesafe_sdk(profile, scene)
        if not jev_res and not self._simple_jev_dead:
            jev_res = self._query_simple_jev(profile, scene)
        if not jev_res and not self._classifier_dead:
            jev_res = self._query_classifier_dev(profile, scene)

        if laya_res and jev_res:
            if laya_res.get("action") == jev_res.get("action"):
                fused = {
                    "action": laya_res.get("action"),
                    "threat_score": max(laya_res.get("threat_score", 0), jev_res.get("threat_score", 0)),
                    "confidence": min(0.99, round((laya_res.get("confidence", 0.8) + jev_res.get("confidence", 0.8)) / 2 + 0.1, 2)),
                    "latency_ms": round(laya_res.get("latency_ms", 15) + jev_res.get("latency_ms", 50), 1),
                    "source": "laya_jev_consensus",
                    "target_coords": laya_res.get("target_coords") or jev_res.get("target_coords"),
                }
            else:
                fused = self._resolve_disagreement(laya_res, jev_res, scene)
            with self._lock:
                self.last_decision = fused
            return fused

        winner = laya_res or jev_res
        if winner:
            with self._lock:
                self.last_decision = winner
        return winner

    def query_jev_async(
        self, profile: GameProfile, scene: UniversalSceneState
    ):
        """Queues scene evaluation asynchronously."""
        if not self._work_queue.full():
            try:
                self._work_queue.put_nowait((profile, scene))
            except queue.Full:
                pass

    def decide(
        self, scene: UniversalSceneState, profile: Optional[GameProfile] = None
    ) -> Dict[str, Any]:
        """Direct synchronous scene evaluation returning actionable decision dictionary."""
        if profile is None:
            from profile_manager import ProfileManager
            profile = ProfileManager().get_profile("mobile_universal")
        return self.get_action(profile, scene)

    def _build_verbal_state(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> str:
        """
        Translates raw telemetry into semantic verbal descriptions.
        Per Laya integration principles: 'Put the state into words, never numbers.'
        """
        parts = [f"Game: {profile.name} ({profile.category})."]

        phase = (getattr(scene, "game_phase", "") or "").strip()
        if phase and phase not in {"active"}:
            parts.append(f"Game phase is {phase}.")

        urgency = float(getattr(scene, "threat_urgency", 0.0) or 0.0)
        if urgency >= 0.85:
            parts.append("Immediate danger is critical — react now.")
        elif urgency >= 0.60:
            parts.append("Danger is elevated — be ready to act.")
        elif urgency >= 0.30:
            parts.append("Some threat present on screen.")
        elif scene.threats:
            parts.append("Minor threats visible but not critical.")

        if scene.threats:
            parts.append(f"Threat count on screen: {len(scene.threats)}.")

        rec = (getattr(scene, "recommended_action", "") or "").strip()
        if rec and rec not in {"wait", "maintain_course", "stand_idle"}:
            parts.append(f"Vision recommends action: {rec}.")

        if hasattr(scene, "elixir") and profile.id.startswith(("mobile_clash", "clash")):
            parts.append(f"Player elixir bank is about {int(scene.elixir)}.")

        # Player telemetry
        if scene.player:
            player_state = "grounded running forward"
            if abs(scene.player.vy) > 80:
                player_state = "airborne jumping" if scene.player.vy < 0 else "falling downwards"
            parts.append(f"Player is {player_state}.")

        # Threat telemetry
        if scene.nearest_threat:
            t = scene.nearest_threat
            # Evaluate proximity verbally
            dist = t.distance_to_player
            if dist <= 110:
                prox = "critical imminent collision range"
            elif dist <= 220:
                prox = "approaching at close range"
            elif dist <= 400:
                prox = "visible on moderate horizon"
            else:
                prox = "distant, plenty of reaction time"

            # Evaluate altitude verbally
            p_bottom = (scene.player.y + scene.player.h) if scene.player else 300
            p_top = scene.player.y if scene.player else 240
            t_bottom = t.y + t.h
            t_center_y = t.y + t.h // 2

            if t_bottom >= p_bottom - 12:
                altitude = "low ground-level obstacle directly in foot path that must be jumped over"
            elif t_center_y <= p_top + 15 and t_bottom > p_top - 25:
                altitude = "mid-height overhead hazard at eye level requiring ducking or sliding underneath"
            elif t_bottom <= p_top - 20:
                altitude = "high overhead obstacle, safe to pass underneath without action"
            else:
                altitude = "obstacle approaching directly in forward path"

            parts.append(f"Obstacle is at {prox}, positioned as {altitude}.")

            # 3-Lane specific verbalization (per skill: 'blocked by a barrier' separates best)
            if profile.id == "runner_3lane":
                lane_desc = self._get_3lane_verbal_status(scene)
                parts.append(lane_desc)
        else:
            parts.append("The path ahead is completely clear with no oncoming threats.")

        # Target telemetry (aim games)
        if scene.best_target:
            parts.append(f"High-contrast target pinpointed in crosshairs at ({scene.best_target.click_x}, {scene.best_target.click_y}).")

        # 8 Ball Pool telemetry
        if "8ball" in profile.id or "pool" in profile.id:
            if getattr(scene, "balls_moving", False):
                parts.append("Balls are currently moving across the table felt. Player must wait until all balls come to a complete standstill.")
            elif not getattr(scene, "cue_ready", False):
                parts.append("The cue stick is not positioned or it is the opponent's turn. Player must wait.")
            elif getattr(scene, "game_phase", "") == "aiming" and getattr(scene, "target_coords", None):
                parts.append(
                    f"Cue stick is ready and aligned. Target ghost ball is at {scene.target_coords} with a clean {scene.cut_angle_deg} degree cut angle. "
                    f"Recommended shot power is {int(scene.shot_power * 100)}%. Ready to execute shot."
                )
            else:
                parts.append("Opening break rack or table layout ready. Ready to execute opening break shot.")

        return " ".join(parts)

    def _get_3lane_verbal_status(self, scene: UniversalSceneState) -> str:
        """Determines lane blockage semantics for 3-lane mobile runners."""
        p_x = scene.player.x if scene.player else 200
        lane_thresh = max(45, int(p_x * 0.25))
        lanes = {"left": "clear", "center": "clear", "right": "clear"}

        for t in scene.threats[:3]:
            rel_x = t.x - p_x
            if rel_x < -lane_thresh:
                lanes["left"] = "blocked by a barrier"
            elif rel_x > lane_thresh:
                lanes["right"] = "blocked by a barrier"
            else:
                lanes["center"] = "blocked by a barrier"

        return f"Lane status: Center lane is {lanes['center']}. Left lane is {lanes['left']}. Right lane is {lanes['right']}."

    def _query_local_laya(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """Inference via local in-process Laya agent."""
        t0 = time.perf_counter()
        try:
            situation = self._build_verbal_state(profile, scene)
            criteria = {
                action.name: action.description for action in profile.actions
            }
            if "wait" not in criteria and profile.category != "clicker":
                criteria["wait"] = "Safe horizon or no immediate movement required"

            questions = {
                "tactical_action": {
                    "type": "choice",
                    "instructions": (
                        f"What is the required tactical maneuver for the player in {profile.name}? "
                        f"When threat is elevated, choose an active response over waiting."
                    ),
                    "criteria": criteria,
                },
                "threat_urgency": {
                    "type": "score",
                    "instructions": "Rate immediate collision risk and reaction urgency from 0 to 4",
                    "criteria": [
                        "0: Safe - Horizon clear",
                        "1: Low - Obstacle distant",
                        "2: Moderate - Obstacle approaching action perimeter",
                        "3: High - Critical reaction threshold",
                        "4: Critical - Imminent impact",
                    ],
                },
                "emergency_reflex": {
                    "type": "noul",
                    "instructions": "Is an evasive maneuver required immediately?",
                },
            }

            result = self.laya_agent.predict(situation, questions)
            latency = (time.perf_counter() - t0) * 1000.0

            ans = result["answers"]
            act_data = ans["tactical_action"]
            chosen_action = act_data["choice"]
            conf = act_data["confidence"]
            urgency_score = min(1.0, float(ans["threat_urgency"]["score"]) / 4.0)

            noul = ans.get("emergency_reflex")
            if isinstance(noul, dict):
                noul = noul.get("noul", noul.get("yes", None))
            if noul is not None:
                try:
                    urgency_score = max(urgency_score, min(1.0, float(noul)))
                except (TypeError, ValueError):
                    pass

            target_coords = None
            if scene.best_target:
                target_coords = (scene.best_target.click_x, scene.best_target.click_y)

            return {
                "action": chosen_action,
                "threat_score": round(urgency_score, 2),
                "confidence": round(conf, 3),
                "latency_ms": round(latency, 1),
                "source": "local_laya_brain",
                "target_coords": target_coords,
            }
        except Exception as e:
            return None

    def _query_local_llm2jev(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """Local LLM2Jev Transformers backend (same Choice/Score/Noul shape as cloud).

        CPU eval is ~50-70s for a full question set — non-blocking for the pilot
        (async worker only). Single-flight: concurrent calls return None immediately.
        """
        engine = self.llm2jev_engine
        if engine is None:
            return None
        if not self._llm2jev_lock.acquire(blocking=False):
            return None
        t0 = time.perf_counter()
        try:
            situation = self._build_verbal_state(profile, scene)
            criteria = {
                action.name: action.description for action in profile.actions
            }
            if "wait" not in criteria and profile.category != "clicker":
                criteria["wait"] = "Hold course; horizon clear"

            questions = {
                "tactical_action": L2JChoice(
                    instructions=(
                        f"Select the optimal immediate maneuver in '{profile.name}' "
                        f"to evade hazards and survive. Prefer concrete action over waiting "
                        f"when threat is elevated."
                    ),
                    criteria=criteria,
                ),
                "threat_severity": L2JScore(
                    instructions="Rate immediate collision urgency on a scale from 0 to 4",
                    criteria=[
                        "0: Safe - Clear horizon",
                        "1: Low - Distant hazard",
                        "2: Moderate - Approaching action threshold",
                        "3: High - Critical reaction zone",
                        "4: Critical - Imminent impact",
                    ],
                ),
                "is_urgent_reflex": L2JNoul(
                    instructions="Should an immediate evasive maneuver be triggered right now?"
                ),
            }
            request = L2JJevRequest(
                state=situation,
                model=self.llm2jev_model,
                questions=questions,
            )
            resp = engine.evaluate(request)
            latency = (time.perf_counter() - t0) * 1000.0

            ans = resp.answers
            action_choice = getattr(
                ans.get("tactical_action"), "choice", None
            ) or profile.actions[0].name
            action_choice = self.coerce_action_name(profile, action_choice, scene)
            action_conf = float(getattr(ans.get("tactical_action"), "confidence", 0.9))
            danger_raw = float(getattr(ans.get("threat_severity"), "score", 2.0))
            danger_normalized = min(1.0, danger_raw / 4.0)

            urgency = getattr(ans.get("is_urgent_reflex"), "noul", None)
            if urgency is not None:
                danger_normalized = max(danger_normalized, min(1.0, float(urgency)))

            target_coords = None
            if scene.best_target:
                target_coords = (scene.best_target.click_x, scene.best_target.click_y)

            return {
                "action": action_choice,
                "threat_score": round(danger_normalized, 2),
                "confidence": round(action_conf, 3),
                "latency_ms": round(latency, 1),
                "source": "local_llm2jev",
                "target_coords": target_coords,
            }
        except Exception:
            return None
        finally:
            self._llm2jev_lock.release()

    def _query_local_openjev(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """Local openjev (AlexWortega/openjev) typed-decision query.

        Scores each profile action as an entailment hypothesis against the verbal
        state; argmax P(entailment) is the chosen action. Single-flight.
        """
        engine = self.openjev_engine
        if engine is None:
            return None
        if not self._openjev_lock.acquire(blocking=False):
            return None
        t0 = time.perf_counter()
        try:
            situation = self._build_verbal_state(profile, scene)
            options = [a.name for a in profile.actions]
            if not options:
                return None
            criteria = {a.name: a.description or a.name for a in profile.actions}
            if "wait" not in options and profile.category != "clicker":
                options.append("wait")
                criteria["wait"] = "Hold course; horizon clear"

            questions = [
                {
                    "type": "choice",
                    "instructions": (
                        f"In '{profile.name}', select the single best immediate action "
                        f"given the situation. Choose exactly one of the options."
                        "\nAllowed answers and rubric: " + json.dumps(criteria, ensure_ascii=False)
                    ),
                    "options": options,
                },
                {
                    "type": "score",
                    "instructions": "Rate immediate danger from 0 (safe) to 4 (critical)",
                    "options": ["0", "1", "2", "3", "4"],
                },
                {
                    "type": "noul",
                    "instructions": "Is an immediate evasive or corrective action required right now?",
                    "options": ["no", "yes"],
                },
            ]
            answers = engine.decide(situation, questions)
            latency = (time.perf_counter() - t0) * 1000.0

            probs = (answers[0] or {}).get("probabilities") or {}
            if not probs:
                return None
            action_choice = max(probs, key=probs.get)
            action_conf = float(probs.get(action_choice, 0.5))
            action_choice = self.coerce_action_name(profile, action_choice, scene)

            danger_probs = (answers[1] or {}).get("probabilities") or {}
            danger_raw = 0.0
            if danger_probs:
                try:
                    danger_raw = max(danger_probs, key=lambda k: float(danger_probs[k]))
                    danger_raw = float(danger_raw)
                except (TypeError, ValueError):
                    danger_raw = 0.0
            danger_normalized = min(1.0, max(0.0, danger_raw / 4.0))

            urgency = (answers[2] or {}).get("noul")
            if urgency is not None:
                # blend noul yes-prob into threat so fusion sees a single danger signal
                danger_normalized = max(danger_normalized, float(urgency))

            target_coords = None
            if scene.best_target:
                target_coords = (scene.best_target.click_x, scene.best_target.click_y)

            return {
                "action": action_choice,
                "threat_score": round(danger_normalized, 2),
                "confidence": round(action_conf, 3),
                "latency_ms": round(latency, 1),
                "source": "local_openjev",
                "target_coords": target_coords,
            }
        except Exception:
            return None
        finally:
            self._openjev_lock.release()

    def _query_gliner_decide(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """Local GLiNER2.5-Decide label-set classifier (fastino, 340M).

        One forward pass scores tactical_action (profile labels), threat_severity
        (0-4 ordinal), and is_urgent_reflex (yes/no). Single-flight.
        """
        engine = self.gliner_engine
        if engine is None:
            return None
        if not self._gliner_lock.acquire(blocking=False):
            return None
        t0 = time.perf_counter()
        try:
            situation = self._build_verbal_state(profile, scene)
            labels = [a.name for a in profile.actions]
            if not labels:
                return None
            if "wait" not in labels and profile.category != "clicker":
                labels.append("wait")

            schema = ClassificationSchema()
            schema.single("tactical_action", labels)
            schema.single("threat_severity", ["0", "1", "2", "3", "4"])
            schema.single("is_urgent_reflex", ["yes", "no"])
            result = engine.classify(situation, schema)
            latency = (time.perf_counter() - t0) * 1000.0

            action_choice = result.value("tactical_action")
            if not isinstance(action_choice, str):
                return None
            conf_raw = result.confidence("tactical_action")
            action_conf = float(conf_raw) if conf_raw is not None else 0.80
            action_choice = self.coerce_action_name(profile, action_choice, scene)

            danger_raw = result.value("threat_severity")
            try:
                danger_normalized = min(1.0, max(0.0, float(danger_raw) / 4.0))
            except (TypeError, ValueError):
                danger_normalized = scene.threat_urgency

            if str(result.value("is_urgent_reflex")).lower() == "yes":
                danger_normalized = max(danger_normalized, 0.90)

            target_coords = None
            if scene.best_target:
                target_coords = (scene.best_target.click_x, scene.best_target.click_y)

            return {
                "action": action_choice,
                "threat_score": round(danger_normalized, 2),
                "confidence": round(action_conf, 3),
                "latency_ms": round(latency, 1),
                "source": "local_gliner_decide",
                "target_coords": target_coords,
            }
        except Exception:
            return None
        finally:
            self._gliner_lock.release()

    def _bev_healthy(self) -> bool:
        """Cached loopback health probe for the Bev decision API (5s TTL)."""
        now = time.monotonic()
        if now - self._bev_checked_at < 5.0:
            return self._bev_up
        try:
            req = urllib.request.Request(self.bev_url + "/health", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                self._bev_up = resp.status == 200
        except Exception:
            self._bev_up = False
        self._bev_checked_at = now
        return self._bev_up

    def _query_bev(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """Bev (Reza2kn/Bev) SystemOne query over loopback HTTP.

        Scores choice / noul / score fields independently against the verbal
        state; argmax probability is the chosen action. Single-flight with a
        hard 8s ceiling so a slow 27B never blocks the fusion chain.
        """
        if not self._bev_healthy():
            return None
        if not self._bev_lock.acquire(blocking=False):
            return None
        t0 = time.perf_counter()
        try:
            situation = self._build_verbal_state(profile, scene)
            options = [a.name for a in profile.actions]
            if not options:
                return None
            criteria = {a.name: a.description or a.name for a in profile.actions}
            if "wait" not in options and profile.category != "clicker":
                options.append("wait")
                criteria["wait"] = "Hold course; horizon clear"

            body = {
                "model": self.bev_model,
                "state": {"text": situation},
                "questions": {
                    "tactical_action": {
                        "type": "choice",
                        "instructions": (
                            f"Select the single best immediate action in '{profile.name}' "
                            f"given the situation."
                        ),
                        "criteria": criteria,
                    },
                    "danger": {
                        "type": "score",
                        "instructions": "Rate immediate danger from 0 (safe) to 4 (critical)",
                        "criteria": ["0", "1", "2", "3", "4"],
                    },
                    "urgent": {
                        "type": "noul",
                        "instructions": "Is an immediate evasive or corrective action required right now?",
                    },
                },
            }
            req = urllib.request.Request(
                self.bev_url + "/v1/systemone",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            latency = (time.perf_counter() - t0) * 1000.0

            answers = payload.get("answers") or {}
            action_ans = answers.get("tactical_action") or {}
            probs = action_ans.get("probabilities") or {}
            if not probs:
                return None
            action_choice = max(probs, key=probs.get)
            action_conf = float(probs.get(action_choice, 0.5))
            action_choice = self.coerce_action_name(profile, action_choice, scene)

            danger_ans = answers.get("danger") or {}
            danger_raw = float(danger_ans.get("score") or 0.0)
            danger_normalized = min(1.0, max(0.0, danger_raw / 4.0))

            urgency = (answers.get("urgent") or {}).get("noul")
            if urgency is not None:
                danger_normalized = max(danger_normalized, float(urgency))

            target_coords = None
            if scene.best_target:
                target_coords = (scene.best_target.click_x, scene.best_target.click_y)

            return {
                "action": action_choice,
                "threat_score": round(danger_normalized, 2),
                "confidence": round(action_conf, 3),
                "latency_ms": round(latency, 1),
                "source": "local_bev",
                "target_coords": target_coords,
            }
        except Exception:
            # Service died mid-flight — force a re-probe on next call.
            self._bev_up = False
            self._bev_checked_at = 0.0
            return None
        finally:
            self._bev_lock.release()

    def _query_typesafe_sdk(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """Inference via TypeSafe Jev System One cloud SDK."""
        t0 = time.perf_counter()
        try:
            situation = self._build_verbal_state(profile, scene)
            criteria = {
                action.name: action.description for action in profile.actions
            }
            if "wait" not in criteria and profile.category != "clicker":
                criteria["wait"] = "Hold course; horizon clear"

            questions = {
                "tactical_action": Choice(
                    instructions=f"Select the optimal immediate maneuver in '{profile.name}' to evade hazards and survive.",
                    criteria=criteria,
                ),
                "threat_severity": Score(
                    instructions="Rate immediate collision urgency on a scale from 0 to 4",
                    criteria=[
                        "0: Safe - Clear horizon",
                        "1: Low - Distant hazard",
                        "2: Moderate - Approaching action threshold",
                        "3: High - Critical reaction zone",
                        "4: Critical - Imminent impact",
                    ],
                ),
                "is_urgent_reflex": Noul(
                    instructions="Should an immediate evasive maneuver be triggered right now?"
                ),
            }

            resp = self.client.system_one(
                state=situation, model=self.model_name, questions=questions
            )
            latency = (time.perf_counter() - t0) * 1000.0

            ans = resp.answers
            action_choice = getattr(ans["tactical_action"], "choice", profile.actions[0].name)
            action_conf = getattr(ans["tactical_action"], "confidence", 0.95)
            danger_raw = float(getattr(ans["threat_severity"], "score", 2.0))
            danger_normalized = min(1.0, danger_raw / 4.0)

            target_coords = None
            if scene.best_target:
                target_coords = (scene.best_target.click_x, scene.best_target.click_y)

            return {
                "action": action_choice,
                "threat_score": round(danger_normalized, 2),
                "confidence": round(action_conf, 3),
                "latency_ms": round(latency, 1),
                "source": "jev_system_one_cloud",
                "target_coords": target_coords,
            }
        except Exception:
            return None

    def _query_classifier_dev(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """Keyless fallback directly using classifier.dev (circuit-broken after hard fail)."""
        if self._classifier_dead:
            return None
        t0 = time.perf_counter()
        try:
            labels = [action.name for action in profile.actions]
            if "wait" not in labels:
                labels.append("wait")

            situation = self._build_verbal_state(profile, scene)
            payload = {
                "labels": labels,
                "input": situation,
                "instructions": (
                    f"Select the single best immediate action for {profile.name}. "
                    "Vault or jump over ground hazards, duck under mid-air obstacles, shift lanes to open spots, or wait if clear."
                ),
            }

            req = urllib.request.Request(
                "https://classifier.dev",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "jev-gamepilot/2.0",
                },
            )

            with urllib.request.urlopen(req, timeout=1.8) as resp:
                data = json.load(resp)

            action = data.get("label", "wait")
            conf = data.get("confidence", 0.90)
            latency = (time.perf_counter() - t0) * 1000

            target_coords = None
            if scene.best_target:
                target_coords = (scene.best_target.click_x, scene.best_target.click_y)

            return {
                "action": action,
                "threat_score": scene.threat_urgency,
                "confidence": round(conf, 3),
                "latency_ms": round(latency, 1),
                "source": "jev_classifier_dev",
                "target_coords": target_coords,
            }
        except urllib.error.HTTPError as e:
            if getattr(e, "code", None) in (403, 404, 429, 500, 502, 503):
                self._classifier_dead = True
            return None
        except Exception:
            return None

    def _query_simple_jev(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """Simple Jev Featherless classifier API (Choice/Score/Noul over HTTP).

        Keyless public demo by default; set FEATHERLESS_API_KEY for production
        endpoint. Circuit-broken on hard HTTP failures; ≥0.55s between calls
        (demo rate limit 2 rps).
        """
        if self._simple_jev_dead:
            return None
        now = time.monotonic()
        if now - self._simple_jev_last_at < 0.55:
            return None
        self._simple_jev_last_at = now
        t0 = time.perf_counter()
        try:
            situation = self._build_verbal_state(profile, scene)
            criteria = {
                action.name: action.description or action.name
                for action in profile.actions
            }
            if "wait" not in criteria and profile.category != "clicker":
                criteria["wait"] = "Hold course; horizon clear"

            payload = {
                "model": self.simple_jev_model,
                "questions": {
                    "tactical_action": {
                        "type": "choice",
                        "instructions": (
                            f"Select the optimal immediate maneuver in '{profile.name}'. "
                            "Prefer concrete action over waiting when threat is elevated."
                        ),
                        "criteria": criteria,
                    },
                    "threat_severity": {
                        "type": "score",
                        "instructions": "Rate immediate collision urgency on a scale from 0 to 4",
                        "criteria": [
                            "0: Safe - Clear horizon",
                            "1: Low - Distant hazard",
                            "2: Moderate - Approaching action threshold",
                            "3: High - Critical reaction zone",
                            "4: Critical - Imminent impact",
                        ],
                    },
                    "is_urgent_reflex": {
                        "type": "noul",
                        "instructions": "Should an immediate evasive maneuver be triggered right now?",
                    },
                },
                "state": situation,
            }
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "jev-gamepilot/2.0",
            }
            if self.simple_jev_key:
                headers["Authorization"] = f"Bearer {self.simple_jev_key}"

            req = urllib.request.Request(
                self.simple_jev_url + "/v1/classifier",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.load(resp)

            answers = data.get("answers") or {}
            act = answers.get("tactical_action") or {}
            action_choice = act.get("choice") or profile.actions[0].name
            action_choice = self.coerce_action_name(profile, action_choice, scene)
            action_conf = float(act.get("confidence") or 0.85)

            sev = answers.get("threat_severity") or {}
            danger_normalized = min(1.0, max(0.0, float(sev.get("score") or 0.0) / 4.0))
            reflex = answers.get("is_urgent_reflex") or {}
            urgency = reflex.get("noul")
            if urgency is not None:
                danger_normalized = max(danger_normalized, min(1.0, float(urgency)))

            latency = (time.perf_counter() - t0) * 1000
            target_coords = None
            if scene.best_target:
                target_coords = (scene.best_target.click_x, scene.best_target.click_y)

            return {
                "action": action_choice,
                "threat_score": round(danger_normalized, 2),
                "confidence": round(action_conf, 3),
                "latency_ms": round(latency, 1),
                "source": "jev_simple_jev",
                "target_coords": target_coords,
            }
        except urllib.error.HTTPError as e:
            if getattr(e, "code", None) in (401, 403, 404, 429, 500, 502, 503):
                self._simple_jev_dead = True
            return None
        except Exception:
            return None

    def _evaluate_intelligent_reflex(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Optional[Dict[str, Any]]:
        """
        Smart spatial & aerodynamic reflex actuator.
        Replaces crude hardcoded default with physical and topological awareness:
        - Detects ground vs mid-air vs overhead obstacles.
        - Detects open lane in 3-lane runners.
        - Evaluates Flappy Bird gap height vs bird fall velocity.
        """
        # 1. Target Clicker / Fruit Ninja Slicing
        if ("fruit" in profile.id or (profile.category == "clicker" and any(a.name == "click_target" for a in profile.actions))) and scene.best_target:
            action_name = (
                ("combo_slice" if len(scene.targets) >= 2 else "slice_target")
                if "fruit" in profile.id else "click_target"
            )
            return {
                "action": action_name,
                "threat_score": 0.95,
                "confidence": 0.99,
                "latency_ms": 0.2,
                "source": "aim_slice_reflex",
                "target_coords": (
                    scene.best_target.click_x,
                    scene.best_target.click_y,
                ),
            }

        # 1b. Solar Smash Planetary Superweapons (Continuous Destruction Reflex)
        if "solar" in profile.id:
            self._solar_strike_step = getattr(self, "_solar_strike_step", 0) + 1
            if self._solar_strike_step % 3 == 0:
                act = "launch_meteor"
            elif self._solar_strike_step % 2 == 0:
                act = "orbital_strike"
            else:
                act = "fire_laser"
            return {
                "action": act,
                "threat_score": 0.90,
                "confidence": 0.98,
                "latency_ms": 0.2,
                "source": "planetary_destruction_reflex",
                "target_coords": None,
            }

        # 1c. Solitaire & Classic Card Puzzles (Continuous Solution Cascade)
        if "solitaire" in profile.id:
            self._sol_step = getattr(self, "_sol_step", 0) + 1
            sol_actions = [
                "sweep_all_columns",
                "tap_col_1", "tap_col_2", "tap_col_3",
                "tap_col_4", "tap_col_5", "tap_col_6", "tap_col_7",
                "tap_waste_card",
                "draw_stock",
                "auto_complete",
            ]
            act = sol_actions[self._sol_step % len(sol_actions)]
            return {
                "action": act,
                "threat_score": 0.10,
                "confidence": 0.95,
                "latency_ms": 0.2,
                "source": "solitaire_card_solver",
                "target_coords": None,
            }

        # 1d. Mobile & PC Card Battlers, TCGs & Deckbuilders (Marvel SNAP, Pokémon Pocket, Hearthstone, Balatro, Slay the Spire)
        if "balatro" in profile.id:
            self._balatro_step = getattr(self, "_balatro_step", 0) + 1
            bal_actions = [
                "select_card",
                "select_card",
                "select_card",
                "balatro_play_hand",
                "select_card",
                "balatro_discard",
                "cash_out",
            ]
            act = bal_actions[self._balatro_step % len(bal_actions)]
            return {
                "action": act,
                "threat_score": 0.40,
                "confidence": 0.95,
                "latency_ms": 0.2,
                "source": "balatro_poker_reflex",
                "target_coords": None,
            }

        if "spire" in profile.id:
            self._spire_step = getattr(self, "_spire_step", 0) + 1
            spire_actions = [
                "play_card_enemy",
                "play_card_enemy",
                "play_card_self",
                "end_turn",
            ]
            act = spire_actions[self._spire_step % len(spire_actions)]
            return {
                "action": act,
                "threat_score": 0.50,
                "confidence": 0.94,
                "latency_ms": 0.2,
                "source": "spire_tactical_reflex",
                "target_coords": None,
            }

        if "card" in profile.id or "hearthstone" in profile.id:
            self._card_step = getattr(self, "_card_step", 0) + 1
            card_actions = [
                "play_card_center",
                "play_card_left",
                "play_card_right",
                "attack_face",
                "hero_power",
                "end_turn",
            ]
            act = card_actions[self._card_step % len(card_actions)]
            return {
                "action": act,
                "threat_score": 0.50,
                "confidence": 0.94,
                "latency_ms": 0.2,
                "source": "tcg_battle_tactics",
                "target_coords": None,
            }

        # 1e. Snake & Grid Arcades (4-Way Navigation Reflex)
        if "snake" in profile.id:
            if scene.best_target and scene.player:
                dx = scene.best_target.click_x - scene.player.x
                dy = scene.best_target.click_y - scene.player.y
                if abs(dx) > abs(dy):
                    act = "turn_right" if dx > 0 else "turn_left"
                else:
                    act = "turn_down" if dy > 0 else "turn_up"
            else:
                self._snake_step = getattr(self, "_snake_step", 0) + 1
                snake_rot = ["turn_right", "turn_down", "turn_left", "turn_up"]
                act = snake_rot[self._snake_step % len(snake_rot)]
            return {
                "action": act,
                "threat_score": 0.70,
                "confidence": 0.96,
                "latency_ms": 0.2,
                "source": "snake_grid_navigation",
                "target_coords": None,
            }

        # 1f. Roblox & Minecraft 3D Action/Parkour
        if "roblox" in profile.id or "minecraft" in profile.id:
            if scene.nearest_threat and scene.threat_urgency > 0.60:
                act = "jump"
            elif len(scene.threats) > 0:
                act = "attack_mine" if "minecraft" in profile.id else "move_left"
            else:
                act = "move_forward"
            return {
                "action": act,
                "threat_score": scene.threat_urgency,
                "confidence": 0.95,
                "latency_ms": 0.2,
                "source": "sandbox_exploration_reflex",
                "target_coords": None,
            }

        # 1g. Trackmania & Racing Reflex
        if "trackmania" in profile.id or "racing" in profile.id:
            if scene.nearest_threat:
                act = "turn_left" if scene.nearest_threat.x > 300 else "turn_right"
            else:
                act = "accelerate"
            return {
                "action": act,
                "threat_score": scene.threat_urgency,
                "confidence": 0.96,
                "latency_ms": 0.2,
                "source": "racing_apex_reflex",
                "target_coords": None,
            }

        # 2. Clash Royale & Tower RTS reflex
        if profile.id == "mobile_clash_royale":
            phase = getattr(scene, "game_phase", "")
            if phase == "matchmaking":
                if self.clash_adapter is not None and self.clash_adapter.learner.battle_open:
                    self.clash_adapter.note_battle_end(None)
                self._clash_was_in_battle = False
                self._clash_battle_clock_armed = False
                return {
                    "action": "wait",
                    "threat_score": 0.0,
                    "confidence": 0.99,
                    "latency_ms": 0.2,
                    "source": "matchmaking_standby",
                    "target_coords": None,
                }
            if phase == "main_menu":
                if self.clash_adapter is not None and self.clash_adapter.learner.battle_open:
                    self.clash_adapter.note_battle_end(getattr(scene, "raw_frame", None))
                self._clash_was_in_battle = False
                self._clash_battle_clock_armed = False
                return {
                    "action": "start_battle",
                    "threat_score": 0.0,
                    "confidence": 0.99,
                    "latency_ms": 0.2,
                    "source": "auto_queue_match",
                    "target_coords": None,
                }
            elif phase == "game_over":
                # Journal + maybe retune BEFORE clearing the battle flags.
                if self.clash_adapter is not None and self.clash_adapter.learner.battle_open:
                    self.clash_adapter.note_battle_end(getattr(scene, "raw_frame", None))
                self._clash_was_in_battle = False
                self._clash_battle_clock_armed = False
                return {
                    "action": "confirm_ok",
                    "threat_score": 0.0,
                    "confidence": 0.99,
                    "latency_ms": 0.2,
                    "source": "post_game_dismiss",
                    "target_coords": None,
                }
            else:  # in_battle
                # 3-Tier Hierarchical Battle Intelligence Pipeline from clash-jev
                if self.clash_adapter is not None:
                    # Reset match clock only on a real menu/queue -> battle edge.
                    # A one-frame false main_menu blip must NOT restart opening tempo.
                    if not getattr(self, "_clash_was_in_battle", False):
                        self._clash_was_in_battle = True
                        if not getattr(self, "_clash_battle_clock_armed", False):
                            self.clash_adapter.note_battle_start()
                            self._clash_battle_clock_armed = True
                    # Stale-clock force-close: a multi-hour open battle is a stuck
                    # start_time (hysteresis kept in_battle after real match ended).
                    # Close as aborted, clear edge flags so next menu→battle re-arms.
                    if (
                        self.clash_adapter.learner.battle_open
                        and (time.time() - self.clash_adapter.start_time) > 600.0
                    ):
                        self.clash_adapter.note_battle_end(getattr(scene, "raw_frame", None))
                        self._clash_was_in_battle = False
                        self._clash_battle_clock_armed = False
                        return {
                            "action": "wait",
                            "threat_score": 0.0,
                            "confidence": 0.99,
                            "latency_ms": 0.2,
                            "source": "stale_battle_force_closed",
                            "target_coords": None,
                        }
                    raw_frame = getattr(scene, "raw_frame", None)
                    if raw_frame is not None:
                        h, w = raw_frame.shape[:2]
                        self.clash_adapter.update_resolution(w, h)
                    else:
                        raw_frame = np.zeros((2340, 1080, 3), dtype=np.uint8)

                    current_elixir = getattr(scene, "elixir", 5)
                    decision = self.clash_adapter.decide(
                        raw_frame,
                        threat_urgency=scene.threat_urgency,
                        current_elixir=current_elixir,
                        threats=scene.threats,
                    )
                    # Adapter derives urgency from lane counts; mirror it back so
                    # phone_pilot NO_PLAY / idle gates see real contact pressure.
                    _ts = float(decision.get("threat_score") or 0.0)
                    if _ts > float(getattr(scene, "threat_urgency", 0.0) or 0.0):
                        scene.threat_urgency = _ts
                    return decision
                else:
                    # Adapter missing — never blind-deploy while elixir strategy says hold.
                    current_elixir = getattr(scene, "elixir", 5)
                    now = time.time()
                    self._last_clash_deploy = getattr(self, "_last_clash_deploy", 0.0)

                    if current_elixir < 3 and (now - self._last_clash_deploy < 2.0):
                        return {
                            "action": "wait",
                            "threat_score": scene.threat_urgency,
                            "confidence": 0.98,
                            "latency_ms": 0.2,
                            "source": "elixir_recharge_standby",
                            "target_coords": None,
                        }

                    if scene.threat_urgency < 0.40 and current_elixir < 6:
                        return {
                            "action": "wait",
                            "threat_score": scene.threat_urgency,
                            "confidence": 0.96,
                            "latency_ms": 0.2,
                            "source": "rts_hold_low_elixir",
                            "target_coords": None,
                        }

                    self._last_clash_deploy = now
                    if scene.nearest_threat and scene.threat_urgency > 0.40:
                        act = "deploy_defense_center"
                    else:
                        self._clash_push_step = getattr(self, "_clash_push_step", 0) + 1
                        act = "deploy_card_left" if (self._clash_push_step % 2 == 0) else "deploy_card_right"

                    return {
                        "action": act,
                        "threat_score": scene.threat_urgency,
                        "confidence": 0.94,
                        "latency_ms": 0.3,
                        "source": "rts_defense_reflex",
                        "target_coords": None,
                    }

        # 2b. Clash of Clans raid reflex
        if profile.id == "mobile_coc":
            phase = getattr(scene, "game_phase", "")
            if phase == "main_menu" or phase == "home_village":
                if self.coc_adapter is not None and self.coc_adapter.raid_open:
                    self.coc_adapter.note_raid_end(getattr(scene, "raw_frame", None))
                self._coc_was_raiding = False
                return {
                    "action": "find_match",
                    "threat_score": 0.0,
                    "confidence": 0.99,
                    "latency_ms": 0.2,
                    "source": "coc_find_match",
                    "target_coords": None,
                }
            if phase == "attack_search":
                if self.coc_adapter is not None and self.coc_adapter.raid_open:
                    self.coc_adapter.note_raid_end(getattr(scene, "raw_frame", None))
                self._coc_was_raiding = False
                return {
                    "action": "wait",
                    "threat_score": 0.0,
                    "confidence": 0.99,
                    "latency_ms": 0.2,
                    "source": "coc_search_standby",
                    "target_coords": None,
                }
            if phase == "results":
                if self.coc_adapter is not None and self.coc_adapter.raid_open:
                    self.coc_adapter.note_raid_end(getattr(scene, "raw_frame", None))
                self._coc_was_raiding = False
                return {
                    "action": "confirm_ok",
                    "threat_score": 0.0,
                    "confidence": 0.99,
                    "latency_ms": 0.2,
                    "source": "coc_loot_dismiss",
                    "target_coords": None,
                }
            # in_raid (or sticky battle evidence)
            if self.coc_adapter is not None:
                if not getattr(self, "_coc_was_raiding", False):
                    self._coc_was_raiding = True
                    if not self.coc_adapter.raid_open:
                        self.coc_adapter.note_raid_start()
                raw_frame = getattr(scene, "raw_frame", None)
                if raw_frame is not None:
                    hh, ww = raw_frame.shape[:2]
                    self.coc_adapter.update_resolution(ww, hh)
                else:
                    raw_frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
                decision = self.coc_adapter.decide(
                    raw_frame,
                    threat_urgency=scene.threat_urgency,
                    phase=getattr(scene, "coc_phase", "") or phase or "in_raid",
                )
                return decision
            return {
                "action": "wait",
                "threat_score": 0.0,
                "confidence": 0.5,
                "latency_ms": 0.2,
                "source": "coc_adapter_missing",
                "target_coords": None,
            }

        # 2c. Brawl Stars match reflex
        if profile.id == "mobile_brawlstars":
            phase = getattr(scene, "game_phase", "")
            if phase == "main_menu" or phase == "menu":
                if self.brawl_adapter is not None and self.brawl_adapter.match_open:
                    self.brawl_adapter.note_match_end(getattr(scene, "raw_frame", None))
                self._brawl_was_in_match = False
                return {
                    "action": "start_battle",
                    "threat_score": 0.0,
                    "confidence": 0.99,
                    "latency_ms": 0.2,
                    "source": "brawl_queue",
                    "target_coords": None,
                }
            if phase == "matchmaking":
                if self.brawl_adapter is not None and self.brawl_adapter.match_open:
                    self.brawl_adapter.note_match_end(getattr(scene, "raw_frame", None))
                self._brawl_was_in_match = False
                return {
                    "action": "wait",
                    "threat_score": 0.0,
                    "confidence": 0.99,
                    "latency_ms": 0.2,
                    "source": "brawl_matchmaking_standby",
                    "target_coords": None,
                }
            if phase == "results":
                if self.brawl_adapter is not None and self.brawl_adapter.match_open:
                    self.brawl_adapter.note_match_end(getattr(scene, "raw_frame", None))
                self._brawl_was_in_match = False
                return {
                    "action": "confirm_ok",
                    "threat_score": 0.0,
                    "confidence": 0.99,
                    "latency_ms": 0.2,
                    "source": "brawl_results_dismiss",
                    "target_coords": None,
                }
            if self.brawl_adapter is not None:
                if not getattr(self, "_brawl_was_in_match", False):
                    self._brawl_was_in_match = True
                    if not self.brawl_adapter.match_open:
                        self.brawl_adapter.note_match_start()
                raw_frame = getattr(scene, "raw_frame", None)
                if raw_frame is not None:
                    hh, ww = raw_frame.shape[:2]
                    self.brawl_adapter.update_resolution(ww, hh)
                else:
                    raw_frame = np.zeros((1080, 2340, 3), dtype=np.uint8)
                decision = self.brawl_adapter.decide(
                    raw_frame,
                    threat_urgency=scene.threat_urgency,
                    phase=getattr(scene, "brawl_phase", "") or phase or "in_match",
                )
                return decision
            return {
                "action": "wait",
                "threat_score": 0.0,
                "confidence": 0.5,
                "latency_ms": 0.2,
                "source": "brawl_adapter_missing",
                "target_coords": None,
            }

        # 3. Earn to Die 2 / 2D Vehicle Driver
        if profile.id == "mobile_earntodie2":
            if len(scene.threats) >= 2 or scene.threat_urgency > 0.85:
                return {
                    "action": "boost",
                    "threat_score": scene.threat_urgency,
                    "confidence": 0.95,
                    "latency_ms": 0.2,
                    "source": "vehicle_nitro_reflex",
                    "target_coords": None,
                }
            return {
                "action": "accelerate",
                "threat_score": scene.threat_urgency,
                "confidence": 0.90,
                "latency_ms": 0.2,
                "source": "vehicle_throttle_reflex",
                "target_coords": None,
            }

        # 4. EA Sports FC / FIFA Mobile (Landscape Pro Controls)
        if profile.id == "mobile_fifa":
            self._fifa_step = getattr(self, "_fifa_step", 0) + 1
            if scene.threat_urgency > 0.65 or len(scene.threats) > 0:
                act = "sprint_tackle"
            elif self._fifa_step % 9 == 0:
                act = "finesse_shot"
            elif self._fifa_step % 8 == 0:
                act = "shoot_goal"
            elif self._fifa_step % 6 == 0:
                act = "skill_move"
            elif self._fifa_step % 5 == 0:
                act = "dribble_cut_inside"
            elif self._fifa_step % 4 == 0:
                act = "dribble_forward"
            elif self._fifa_step % 3 == 0:
                act = "through_pass"
            elif self._fifa_step % 2 == 0:
                act = "pass"
            else:
                act = "sprint_tackle"
            return {
                "action": act,
                "threat_score": scene.threat_urgency,
                "confidence": 0.95,
                "latency_ms": 0.2,
                "source": "fifa_pro_controls",
                "target_coords": None,
            }

        # 5. BitLife & Choice Simulators
        if profile.id == "mobile_bitlife":
            act = "primary_choice" if scene.targets else "age_up"
            return {
                "action": act,
                "threat_score": 0.0,
                "confidence": 0.96,
                "latency_ms": 0.2,
                "source": "bitlife_progression_reflex",
                "target_coords": None,
            }

        # 6. Imminent Hazard Collision (Strike Zone) for Runners & Platformers
        if scene.threat_urgency >= 0.70 and scene.nearest_threat:
            t = scene.nearest_threat
            p_bottom = (scene.player.y + scene.player.h) if scene.player else 300
            p_top = scene.player.y if scene.player else 240
            t_bottom = t.y + t.h
            t_center_y = t.y + t.h // 2

            # Find actions by intention
            duck_action = next(
                (a.name for a in profile.actions if "duck" in a.name or "slide" in a.name or a.key == "down"),
                None,
            )
            jump_action = next(
                (a.name for a in profile.actions if "jump" in a.name or "vault" in a.name or a.key in ["space", "up"]),
                None,
            )
            left_action = next(
                (a.name for a in profile.actions if "left" in a.name or a.key in ["left", "a"]),
                None,
            )
            right_action = next(
                (a.name for a in profile.actions if "right" in a.name or a.key in ["right", "d"]),
                None,
            )

            # 3-Lane Runner & Universal Mobile Runner: pick clear lane
            if profile.id in ["runner_3lane", "mobile_universal"] or len(profile.actions) >= 4:
                p_x = scene.player.x if scene.player else 200
                lane_thresh = max(45, int(p_x * 0.25))
                danger_dist = max(180, int((scene.player.h if scene.player else 60) * 3.5))
                rel_x = t.x - p_x
                if abs(rel_x) < lane_thresh:
                    # In front of obstacle: evade to whichever side is open
                    has_left_obstacle = any(o.x - p_x < -lane_thresh and o.distance_to_player < danger_dist for o in scene.threats)
                    has_right_obstacle = any(o.x - p_x > lane_thresh and o.distance_to_player < danger_dist for o in scene.threats)
                    hoverboard_action = next((a.name for a in profile.actions if "hoverboard" in a.name or "shield" in a.name), None)
                    if has_left_obstacle and has_right_obstacle and scene.threat_urgency > 0.82 and hoverboard_action:
                        chosen = hoverboard_action
                    elif not has_left_obstacle and left_action:
                        chosen = left_action
                    elif not has_right_obstacle and right_action:
                        chosen = right_action
                    elif t_bottom < p_bottom - 20 and duck_action:
                        chosen = duck_action
                    else:
                        chosen = jump_action or profile.actions[0].name
                    return {
                        "action": chosen,
                        "threat_score": scene.threat_urgency,
                        "confidence": 0.96,
                        "latency_ms": 0.3,
                        "source": "lane_evasion_reflex",
                        "target_coords": None,
                    }

            # Flappy Bird Tap Profile
            if profile.id == "flappy_tap":
                # If falling or below threat horizon, tap/flap
                flap_action = next((a.name for a in profile.actions if a.key == "space" or "flap" in a.name), profile.actions[0].name)
                glide_action = next((a.name for a in profile.actions if a.key == "none" or "glide" in a.name), "wait")
                if scene.player and scene.player.vy > 10:  # Falling downward
                    return {
                        "action": flap_action,
                        "threat_score": scene.threat_urgency,
                        "confidence": 0.95,
                        "latency_ms": 0.2,
                        "source": "altitude_flap_reflex",
                        "target_coords": None,
                    }
                else:
                    return {
                        "action": glide_action,
                        "threat_score": scene.threat_urgency,
                        "confidence": 0.90,
                        "latency_ms": 0.2,
                        "source": "altitude_glide_reflex",
                        "target_coords": None,
                    }

            # Standard 2D Runner / Platformer (Dino, Retro Platformer):
            # Mid-height hazard (bird/overhead beam) -> Duck!
            # Only when the threat is actually in the player's horizontal path
            # (prevents side-scroll false positives from stealing the lane decision).
            p_x = scene.player.x if scene.player else 0
            p_w = scene.player.w if scene.player else 40
            rel_x = t.x - p_x
            in_player_path = abs(rel_x) <= max(p_w + 24, 64)
            if (
                in_player_path
                and t_center_y <= p_top + 25
                and t_bottom > p_top - 20
                and duck_action
            ):
                return {
                    "action": duck_action,
                    "threat_score": scene.threat_urgency,
                    "confidence": 0.98,
                    "latency_ms": 0.3,
                    "source": "smart_duck_reflex",
                    "target_coords": None,
                }

            # Ground level hazard (cactus/pit/hurdle) -> Jump!
            # Same path gate as duck so off-screen noise never triggers a jump.
            if in_player_path and jump_action:
                return {
                    "action": jump_action,
                    "threat_score": scene.threat_urgency,
                    "confidence": 0.98,
                    "latency_ms": 0.3,
                    "source": "smart_jump_reflex",
                    "target_coords": None,
                }

        return None

    @staticmethod
    def coerce_action_name(
        profile: GameProfile,
        action_name: str,
        scene: Optional[UniversalSceneState] = None,
    ) -> str:
        """Map any brain output onto a name the profile/dispatcher actually understands."""
        name = (action_name or "").strip()
        if not name:
            return "wait"
        idle = {"wait", "maintain_course", "stand_idle"}
        if name in idle:
            return name
        valid = {a.name for a in profile.actions}
        if name in valid:
            return name
        lowered = name.lower()
        for a in profile.actions:
            al = a.name.lower()
            if al == lowered or al in lowered or lowered in al:
                return a.name
        if scene is not None:
            rec = (getattr(scene, "recommended_action", "") or "").strip()
            if rec in valid or rec in idle:
                return rec
        return "wait"

    def _local_scene_decision(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Dict[str, Any]:
        """Sub-ms local answer when cloud cache is stale — never leave the loop on wait/init."""
        rec = (getattr(scene, "recommended_action", "") or "").strip() or "wait"
        urgency = float(getattr(scene, "threat_urgency", 0.0) or 0.0)

        action = self.coerce_action_name(profile, rec, scene)
        if action in {"wait", "maintain_course", "stand_idle"} and urgency >= 0.5:
            # Vision left recommended_action idle but threat is real — derive from geometry.
            if scene.nearest_threat and scene.player:
                t = scene.nearest_threat
                lane_actions = {a.name: a for a in profile.actions}
                left = next((n for n in lane_actions if "left" in n), None)
                right = next((n for n in lane_actions if "right" in n), None)
                duck = next(
                    (n for n in lane_actions if "duck" in n or "slide" in n),
                    None,
                )
                jump = next((n for n in lane_actions if "jump" in n), None)
                rel_x = t.x - scene.player.x
                if abs(rel_x) < max(45, int(scene.player.x * 0.25)) and duck:
                    action = duck
                elif rel_x < 0 and right:
                    action = right
                elif rel_x > 0 and left:
                    action = left
                elif jump:
                    action = jump
                else:
                    action = "maintain_course" if "maintain_course" in {a.name for a in profile.actions} else "wait"
            else:
                action = "maintain_course" if "maintain_course" in {a.name for a in profile.actions} else "wait"

        target_coords = None
        if scene.best_target:
            target_coords = (scene.best_target.click_x, scene.best_target.click_y)
        elif getattr(scene, "target_coords", None):
            target_coords = scene.target_coords

        return {
            "action": action,
            "threat_score": urgency,
            "confidence": 0.72 if action not in {"wait", "maintain_course", "stand_idle"} else 0.9,
            "latency_ms": 0.2,
            "source": "local_scene_heuristic",
            "target_coords": target_coords,
        }

    def get_action(
        self, profile: GameProfile, scene: UniversalSceneState
    ) -> Dict[str, Any]:
        """
        Hybrid decision pipeline:
        1. Instant spatial reflex when hazard is in strike zone / path.
        2. Async cloud refresh queued when threat or targets appear.
        3. Fresh local scene heuristic when cache is stale (init/idle) —
           never block the 60 FPS loop on a 800ms cloud round-trip.
        4. Otherwise reuse the latest high-confidence cloud decision.
        """
        # 1. Smart Physical Reflex (instant sub-ms execution)
        reflex_decision = self._evaluate_intelligent_reflex(profile, scene)
        if reflex_decision is not None:
            coerced = self.coerce_action_name(profile, reflex_decision.get("action", "wait"), scene)
            if coerced != reflex_decision.get("action"):
                reflex_decision = dict(reflex_decision)
                reflex_decision["action"] = coerced
            with self._lock:
                self.last_decision = reflex_decision
            return reflex_decision

        # 2. Queue asynchronous brain evaluation when threats or targets are detected
        if (scene.threat_urgency > 0.12 or scene.targets) and not self._work_queue.full():
            try:
                self._work_queue.put_nowait((profile, scene))
            except queue.Full:
                pass

        with self._lock:
            cached = dict(self.last_decision)

        # 3. Stale cache (init / pure idle) + live scene signal → answer locally now
        cache_source = cached.get("source", "init")
        cache_action = cached.get("action", "wait")
        cache_is_stale = cache_source in (None, "init") or cache_action in {
            "wait",
            "maintain_course",
            "stand_idle",
        }
        scene_rec = (getattr(scene, "recommended_action", "") or "").strip()
        scene_has_signal = (
            scene.threat_urgency > 0.12
            or bool(scene.targets)
            or scene_rec not in {"", "wait", "maintain_course", "stand_idle"}
        )
        if cache_is_stale and scene_has_signal:
            local = self._local_scene_decision(profile, scene)
            with self._lock:
                # Don't clobber a fresher decision the worker may have just written
                if self.last_decision.get("source", "init") in (None, "init"):
                    self.last_decision = local
            return local

        # 4. Reuse recent cloud decision unless it's blind to an urgent threat
        if (
            cached.get("confidence", 0.0) >= 0.5
            and not (
                scene.threat_urgency >= 0.7
                and float(cached.get("threat_score", 0.0) or 0.0) < 0.3
            )
        ):
            if not cached.get("target_coords") and scene.best_target:
                cached["target_coords"] = (scene.best_target.click_x, scene.best_target.click_y)
            return cached

        # 5. Final fallback: local heuristic (never raw wait/init)
        return self._local_scene_decision(profile, scene)
