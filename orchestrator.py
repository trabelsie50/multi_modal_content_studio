import asyncio
import logging
import os
from typing import List, Optional

from config import FINAL_VIDEO_DIR, get_settings, OUTPUT_DIR, SCENES_DIR
from gemini_service import GeminiService, extract_script
from image_service import ImageService, enhance_image, generate_scene_image
from models import ArticleInput, ImageResult, PipelineResult, Script, VideoResult, VoiceoverResult, Scene
from tts_service import TTSService, generate_voiceover
from video_service import VideoService

logger = logging.getLogger(__name__)
settings = get_settings()

# التأكد من وجود أدلة الإخراج قبل بدء المعالجة
for _dir in [OUTPUT_DIR, SCENES_DIR, FINAL_VIDEO_DIR]:
    os.makedirs(_dir, exist_ok=True)


class ContentOrchestrator:
    """المنسق الرئيسي لسير العمل متعدد الوسائط (Multi-AI Orchestration).

    يربط جميع مراحل المعالجة ببعضها البعض:
      1. استقبال المقال وتنقيحه عبر Google Gemini API
      2. توليد وتحسين الصور لكل مشهد
      3. تحويل السيناريو إلى تعليق صوتي عبر ElevenLabs
      4. دمج جميع العناصر في فيديو نهائي عبر MoviePy
    """

    def __init__(
        self,
        gemini_service: Optional[GeminiService] = None,
        image_service: Optional[ImageService] = None,
        tts_service: Optional[TTSService] = None,
        video_service: Optional[VideoService] = None,
    ):
        """تهيئة المنسق مع إمكانية حقن الخدمات (Dependency Injection).

        Args:
            gemini_service: خدمة Gemini لاستخلاص السيناريو. إذا لم تُقدَّم، تُنشأ نسخة افتراضية.
            image_service: خدمة توليد الصور. إذا لم تُقدَّم، تُنشأ نسخة افتراضية.
            tts_service: خدمة تحويل النص إلى صوت. إذا لم تُقدَّم، تُنشأ نسخة افتراضية.
            video_service: خدمة مونتاج الفيديو. إذا لم تُقدَّم، تُنشأ نسخة افتراضية.
        """
        self.gemini_service = gemini_service or GeminiService()
        self.image_service = image_service or ImageService()
        self.tts_service = tts_service or TTSService()
        self.video_service = video_service or VideoService()

    async def execute(self, article_input: ArticleInput) -> PipelineResult:
        """تنفيذ سير العمل الكامل من المدخلات إلى الفيديو النهائي.

        Args:
            article_input: المدخلات الأصلية (رابط مقال أو نص خام).

        Returns:
            PipelineResult يحتوي على جميع نتائج المراحل الأربع.
        """
        logger.info(
            "بدء سير العمل للمقال: %s | المصدر: %s",
            getattr(article_input, "title", ""),
            getattr(article_input, "source", ""),
        )

        # المرحلة 1: الاستقبال والتنقيح — استخراج السيناريو والمشاهد باستخدام Gemini
        script = await self._extract_script(article_input)
        logger.info("تم استخراج السيناريو بنجاح بـ %d مشهد.", len(script.scenes))

        # المرحلة 2: توليد وتحسين الصور لكل مشهد بشكل متوازٍ
        images = await self._generate_images(script.scenes)
        logger.info("تم توليد وتحسين %d صورة بنجاح.", len(images))

        # المرحلة 3: التوليد الصوتي — تحويل نص السيناريو إلى تعليق صوتي
        voiceover = await self._generate_voiceover(script)
        logger.info("تم إنشاء التعليق الصوتي: %s", getattr(voiceover, "filepath", "N/A"))

        # المرحلة 4: الدمج والمونتاج الآلي — تركيب الفيديو النهائي من الصور والصوت
        video_result = await self._assemble_video(images, voiceover, script)
        logger.info("تم تركيب الفيديو النهائي: %s", getattr(video_result, "filepath", "N/A"))

        return PipelineResult(
            script=script,
            images=images,
            voiceover=voiceover,
            video_result=video_result,
        )

    async def _extract_script(self, article_input: ArticleInput) -> Script:
        """المرحلة 1: استخلاص السيناريو والمشاهد القصيرة من المقال أو النص.

        يستدعي Google Gemini API عبر دالة extract_script القرائية لتحليل المحتوى
        واستخراج وصف كل مشهد مع السيناريو العام.

        Args:
            article_input: المدخلات الأصلية (رابط أو نص).

        Returns:
            Script يحتوي على المشاهد المستخرجة والوصف العام.
        """
        return await extract_script(article_input)

    async def _generate_images(self, scenes: List[Scene]) -> List[ImageResult]:
        """المرحلة 2: توليد وتحسين الصور لكل مشهد.

        يتم توليد الصور بشكل متوازٍ باستخدام asyncio.gather لتسريع المعالجة،
        ثم تُحسَّن جودة كل صورة عبر دالة enhance_image لترقية الدقة
        لتناسب الفيديو العمودي (1080x1920).

        Args:
            scenes: قائمة المشاهد المستخرجة من السيناريو.

        Returns:
            قائمة الصور المولدة والمحسّنة لكل مشهد.
        """
        # توليد الصور بشكل متوازي لجميع المشاهد
        raw_images: List[ImageResult] = await asyncio.gather(
            *[generate_scene_image(scene, service=self.image_service) for scene in scenes]
        )

        # تحسين جودة كل صورة مولّدة
        enhanced_images: List[ImageResult] = []
        for raw_image, scene in zip(raw_images, scenes):
            enhanced = enhance_image(raw_image, scene)
            enhanced_images.append(enhanced)

        return enhanced_images

    async def _generate_voiceover(self, script: Script) -> VoiceoverResult:
        """المرحلة 3: تحويل نص السيناريو إلى تعليق صوتي احترافي.

        يستخدم ElevenLabs API لتوليد ملف صوتي MP3 من نص السيناريو،
        مع دعم اختيار الصوت ونبرة الصوت.

        Args:
            script: السيناريو المستخرج من المرحلة الأولى.

        Returns:
            VoiceoverResult يحتوي على مسار الملف الصوتي ومعلومات التعليق.
        """
        return await generate_voiceover(script, service=self.tts_service)

    async def _assemble_video(
        self,
        images: List[ImageResult],
        voiceover: VoiceoverResult,
        script: Script,
    ) -> VideoResult:
        """المرحلة 4: دمج الصور والتعليق الصوتي في فيديو نهائي عمودي.

        يستخدم MoviePy لدمج جميع العناصر: الصور المولّدة، التعليق الصوتي،
        المؤثرات الانتقالية (transitions)، وخلفية موسيقية اختيارية،
        لإنتاج فيديو عمودي (1080x1920) مناسب لـ Reels و TikTok.

        Args:
            images: قائمة الصور المولّدة والمحسّنة.
            voiceover: نتيجة التعليق الصوتي من المرحلة الثالثة.
            script: السيناريو النصي لعرضه كنص على الفيديو.

        Returns:
            VideoResult يحتوي على مسار الفيديو النهائي ومعلوماته.
        """
        return await asyncio.to_thread(
            self.video_service.assemble_video,
            images,
            voiceover,
            script,
            True,       # transitions: تفعيل المؤثرات الانتقالية
            None,       # background_music_path: لا توجد موسيقى خلفية افتراضية
        )


async def run_pipeline(article_input: ArticleInput) -> PipelineResult:
    """نقطة الدخول الرئيسية لتشغيل سير العمل الكامل.

    تنشئ منسق المحتوى (ContentOrchestrator) وتنفذ سير العمل بالكامل
    من المدخلات إلى الفيديو النهائي.

    Args:
        article_input: المدخلات الأصلية (رابط مقال أو نص خام).

    Returns:
        PipelineResult يحتوي على جميع نتائج المراحل:
        - script: السيناريو المستخرج
        - images: الصور المولّدة لكل مشهد
        - voiceover: التعليق الصوتي
        - video_result: الفيديو النهائي الجاهز للنشر
    """
    orchestrator = ContentOrchestrator()
    return await orchestrator.execute(article_input)
