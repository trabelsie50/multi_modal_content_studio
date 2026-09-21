"""
خدمة تحويل النص إلى كلام (Text-to-Speech).
تستخدم ElevenLabs API لتوليد تعليق صوتي احترافي من نص السيناريو المستخرج،
مع دعم لاختيار الصوت ونبرة الصوت. تُنتج ملفاً صوتياً بصيغة MP3 جاهزاً للدمج.
"""

import os
import time
import logging
from typing import Any, Optional

import requests

from config import ELEVENLABS_API_KEY, OUTPUT_DIR
from models import VoiceoverResult, Script

logger = logging.getLogger(__name__)

# ── ثوابت الخدمة ──────────────────────────────────────────────────────────
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
DEFAULT_MODEL_ID = "eleven_monolingual_v1"
DEFAULT_STABILITY = 0.5
DEFAULT_SIMILARITY_BOOST = 0.75
REQUEST_TIMEOUT = 120
MAX_TEXT_CHARS = 5000
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")


class TTSService:
    """خدمة تحويل النص إلى كلام باستخدام ElevenLabs API."""

    def __init__(
        self,
        api_key: str = ELEVENLABS_API_KEY,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL_ID,
        stability: float = DEFAULT_STABILITY,
        similarity_boost: float = DEFAULT_SIMILARITY_BOOST,
        output_dir: str = AUDIO_DIR,
    ) -> None:
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.output_dir = output_dir
        self._ensure_output_dir()

    # ── إعداد المجلد ────────────────────────────────────────────────────────
    def _ensure_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    # ── بناء الطلب ──────────────────────────────────────────────────────────
    def _get_headers(self) -> dict[str, Any]:
        return {
            "Accept": "audio/mpeg",
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        text: str,
        voice_id: str,
        stability: Optional[float] = None,
        similarity_boost: Optional[float] = None,
    ) -> dict[str, Any]:
        return {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": stability if stability is not None else self.stability,
                "similarity_boost": (
                    similarity_boost
                    if similarity_boost is not None
                    else self.similarity_boost
                ),
            },
        }

    # ── استدعاء API ─────────────────────────────────────────────────────────
    def _call_elevenlabs(self, payload: dict[str, Any], voice_id: str) -> bytes:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        logger.info("إرسال طلب TTS إلى ElevenLabs للصوت: %s", voice_id)
        response = requests.post(
            url,
            headers=self._get_headers(),
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        logger.info("استلام رد ناجح من ElevenLabs (%d بايت)", len(response.content))
        return response.content

    # ── حفظ الملف الصوتي ────────────────────────────────────────────────────
    def _save_audio(self, audio_data: bytes, filename: str) -> str:
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(audio_data)
        return filepath

    # ── تقدير مدة الصوت ─────────────────────────────────────────────────────
    @staticmethod
    def _estimate_duration(audio_data: bytes) -> float:
        bitrate_bps = 128000
        if len(audio_data) == 0:
            return 0.0
        duration_seconds = (len(audio_data) * 8) / bitrate_bps
        return round(duration_seconds, 2)

    # ── الطريقة العامة لتوليد التعليق الصوتي ────────────────────────────────
    def generate_voiceover(
        self,
        text: str,
        voice_id: Optional[str] = None,
        stability: Optional[float] = None,
        similarity_boost: Optional[float] = None,
        filename: Optional[str] = None,
    ) -> VoiceoverResult:
        selected_voice = voice_id if voice_id is not None else self.voice_id

        payload = self._build_payload(
            text=text,
            voice_id=selected_voice,
            stability=stability,
            similarity_boost=similarity_boost,
        )

        logger.info(
            "بدء توليد التعليق الصوتي | الصوت: %s | طول النص: %d حرف",
            selected_voice,
            len(text),
        )

        audio_data = self._call_elevenlabs(payload, selected_voice)

        if filename is None:
            timestamp = int(time.time())
            filename = f"voiceover_{timestamp}.mp3"

        filepath = self._save_audio(audio_data, filename)
        duration = self._estimate_duration(audio_data)

        logger.info(
            "تم حفظ التعليق الصوتي | المسار: %s | المدة: %.2f ثانية",
            filepath,
            duration,
        )

        return VoiceoverResult(
            file_path=filepath,
            duration=duration,
            voice_name=selected_voice,
            text_length=len(text),
        )


# ── دالة مساعدة غير متزامنة للاستخدام عبر المنسق ────────────────────────────

async def generate_voiceover(
    script: Script,
    voice_id: str = DEFAULT_VOICE_ID,
    stability: float = DEFAULT_STABILITY,
    similarity_boost: float = DEFAULT_SIMILARITY_BOOST,
    service: Optional[TTSService] = None,
) -> VoiceoverResult:
    """
    دالة مساعدة لتوليد التعليق الصوتي من سيناريو نصي.

    تقوم بتحويل كائن Script إلى نص كامل، ثم تستدعي TTSService لتوليد
    ملف صوتي MP3 جاهز للدمج مع الصور في مرحلة المونتاج.

    Args:
        script: كائن Script يحتوي على المشاهد والنص المراد تحويله إلى صوت.
        voice_id: معرف الصوت المطلوب استخدامه في ElevenLabs.
        stability: استقرار الصوت (0.0 = متغير، 1.0 = ثابت).
        similarity_boost: تعزيز التشابه مع العينة الصوتية الأصلية.
        service: مثيل TTSService مخصص (اختياري؛ يُنشئ مثيلاً جديداً إذا لم يُقدَّم).

    Returns:
        VoiceoverResult يحتوي على مسار الملف الصوتي ومدته وبيانات وصفية.
    """
    if service is None:
        service = TTSService()

    # ── بناء النص الكامل من السيناريو ────────────────────────────────────────
    full_text = _build_text_from_script(script)

    # ── تقطيع النص إذا تجاوز الحد المسموح ───────────────────────────────────
    if len(full_text) > MAX_TEXT_CHARS:
        logger.warning(
            "النص يتجاوز %d حرف، سيتم تقطيعه إلى هذا الحد.", MAX_TEXT_CHARS
        )
        full_text = full_text[:MAX_TEXT_CHARS]

    # ── توليد التعليق الصوتي ─────────────────────────────────────────────────
    return service.generate_voiceover(
        text=full_text,
        voice_id=voice_id,
        stability=stability,
        similarity_boost=similarity_boost,
    )


def _build_text_from_script(script: Script) -> str:
    """يبني نصاً كاملاً من كائن Script عبر تجميع نص كل مشهد."""
    text_parts: list[str] = []

    if script.scenes:
        for scene in script.scenes:
            text_parts.append(f"[المشهد {scene.scene_id}]")
            if scene.description:
                text_parts.append(scene.description)
            if scene.script:
                text_parts.append(scene.script)
    elif script.script:
        text_parts.append(script.script)
    else:
        text_parts.append(str(script))

    return "\n".join(text_parts)
