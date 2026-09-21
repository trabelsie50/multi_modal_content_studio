"""
models.py

تعريف جميع نماذج البيانات (dataclasses) التي تُشكّل اللغة المشتركة
بين جميع مراحل سير العمل في Multi-Modal Content Studio.

يصدّر:
    - ArticleInput
    - Scene
    - Script
    - ImageResult
    - VoiceoverResult
    - VideoResult
    - PipelineResult
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ArticleInput:
    """
    يمثّل المدخل الأصلي من المستخدم: رابط مقال أو نص خام.
    يُمرَّر من البوت إلى مرحلة الاستقبال والاستخلاص (Ingestion).
    """
    source_url: Optional[str] = None
    raw_text: Optional[str] = None
    title: Optional[str] = None


@dataclass
class Scene:
    """
    يمثّل مشهداً واحداً ضمن السيناريو المستخرج.
    يحتوي على الوصف البصري (للتوليد الصوتي) ونص المشهد (للتعليق الصوتي).
    يُمرَّر بين مرحلة Gemini (الاستخلاص) ومرحلة توليد الصور والصوت.
    """
    scene_id: int
    description: str
    script_text: str
    image_prompt: Optional[str] = None
    duration_seconds: int = 5


@dataclass
class Script:
    """
    يمثّل السيناريو الكامل المستخرج من المقال أو الفكرة.
    يتضمن العنوان، ملخصاً عاماً، وقائمة المشاهد (Scenes).
    يُنتَج بواسطة GeminiService ويُستهلك بواسطة ImageService و TTSService.
    """
    title: str
    summary: str
    scenes: List[Scene] = field(default_factory=list)
    total_duration: int = 0


@dataclass
class ImageResult:
    """
    يمثّل نتيجة توليد صورة واحدة لمشهد معيّن.
    يحتوي على المسار المحلي للصورة وأبعادها ودرجة الجودة.
    يُنتَج بواسطة ImageService ويُستهلك بواسطة VideoService.
    """
    scene_id: int
    image_path: str
    image_url: Optional[str] = None
    width: int = 1080
    height: int = 1920
    quality_score: float = 0.0


@dataclass
class VoiceoverResult:
    """
    يمثّل نتيجة توليد التعليق الصوتي (Voiceover) من نص السيناريو.
    يحتوي على مسار الملف الصوتي ومدته والصوت المستخدم.
    يُنتَج بواسطة TTSService ويُستهلك بواسطة VideoService.
    """
    script_text: str
    audio_path: str
    duration_seconds: float = 0.0
    voice_name: str = "default"


@dataclass
class VideoResult:
    """
    يمثّل نتيجة الفيديو النهائي المُركَّب.
    يحتوي على مسار الفيديو وأبعاده ومدته ومعلومات الموسيقى الخلفية.
    يُنتَج بواسطة VideoService ويُرسَل إلى المستخدم عبر البوت.
    """
    video_path: str
    width: int = 1080
    height: int = 1920
    duration_seconds: float = 0.0
    background_music: Optional[str] = None


@dataclass
class PipelineResult:
    """
    يمثّل النتيجة الكاملة لسير العمل (Pipeline).
    يجمع كل مخرجات المراحل: السيناريو، الصور، التعليق الصوتي، والفيديو النهائي.
    يُنتَج بواسطة ContentOrchestrator ويُرسَل إلى البوت للتسليم.
    """
    input_data: ArticleInput
    script: Script
    image_results: List[ImageResult] = field(default_factory=list)
    voiceover_result: Optional[VoiceoverResult] = None
    video_result: Optional[VideoResult] = None
    status: str = "completed"
