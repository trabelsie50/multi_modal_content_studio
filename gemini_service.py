"""
gemini_service.py

خدمة التنقيح والاستخلاص باستخدام Google Gemini API.
تستقبل رابط مقال أو نصاً خاماً، تقرأ محتواه عبر استدعاء النموذج
متعدد الوسائط (gemini-pro-vision)، وتستخرج سيناريو مقسّماً إلى مشاهد
قصيرة (Scenes) مع وصف بصري لكل مشهد.

المرحلة 1 من سير العمل (Ingestion & Extraction).
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

import requests
from google.generativeai import GenerativeModel, types

from config import get_settings
from models import ArticleInput, Scene, Script

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ثوابت الخدمة
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME: str = "gemini-1.5-pro"
FALLBACK_MODEL_NAME: str = "gemini-pro-vision"

# الحد الأقصى لعدد الأحرف في النص المرسل للنموذج في استدعاء واحد
MAX_CONTENT_CHUNK_LENGTH: int = 30_000

# النمط المستخدم لتحليل المشاهد من مخرجات النموذج
SCENE_PATTERN: re.Pattern[str] = re.compile(
    r"(?:Scene\s*\d*[:\-\s]*)?\s*([A-Z][A-Z\s\-]{2,}(?:\s+[A-Z][A-Z\s\-]{2,})*)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# دالة مساعدة لجلب محتوى المقال من رابط URL
# ---------------------------------------------------------------------------
def fetch_article_content(url: str, timeout: int = 30) -> str:
    """
    تقوم بجلب محتوى المقال من الرابط المُقدَّم وإرجاع نصّه.

    تستخدم requests لجلب الصفحة ثم تستخرج النص المرئي باستخدام
    تحليل بسيط لـ HTML (بدون اعتماد خارجي إضافي). في حال فشل الجلب،
    تُرجع سلسلة فارغة.

    المعاملات:
        url: رابط المقال المراد قراءته.
        timeout: مهلة الطلب بالثواني.

    يُرجع:
        نص المقال المستخرج كسلسلة نصية.
    """
    headers: Dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ar,en;q=0.9",
    }

    try:
        response: requests.Response = requests.get(
            url, headers=headers, timeout=timeout
        )
        response.raise_for_status()
        content: str = response.text
        text: str = _extract_text_from_html(content)
        logger.info(
            "تم جلب المقال بنجاح من %s | طول النص المستخرج: %d حرف",
            url,
            len(text),
        )
        return text
    except requests.RequestException as exc:
        logger.warning("فشل جلب المقال من %s: %s", url, exc)
        return ""


def _extract_text_from_html(html: str) -> str:
    """
    تستخرج النص المرئي من مستند HTML باستخدام تعبيرات منتظمة.

    تزيل السكربتات والتنسيقات HTML وتُرجع النص النظيف.

    المعاملات:
        html: كود HTML الخام.

    يُرجع:
        النص المستخرج ومُنظّف.
    """
    # إزالة السكربتات والأنماط
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # إزالة العلامات HTML
    text: str = re.sub(r"<[^>]+>", " ", html)
    # توحيد المسافات البيضاء
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# كلاس GeminiService
# ---------------------------------------------------------------------------
class GeminiService:
    """
    خدمة التنقيح والاستخلاص باستخدام Google Gemini API.

    مسؤولية هذه الخدمة:
        - تهيئة نموذج Gemini المناسب للتحليل متعدد الوسائط.
        - استقبال رابط مقال أو نص خام.
        - قراءة المحتوى واستخراج سيناريو مقسّم إلى مشاهد قصيرة (Scenes)
          مع وصف بصري لكل مشهد.

    المرحلة: 1 (Ingestion & Extraction)
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        يُهيئ خدمة Gemini.

        المعاملات:
            model_name: اسم نموذج Gemini المراد استخدامه.
                        إذا لم يُحدَّد، يُستخدم gemini-1.5-pro مع fallback
                        إلى gemini-pro-vision.
        """
        self._settings = get_settings()
        self._model_name: str = model_name or DEFAULT_MODEL_NAME
        self._fallback_model_name: str = FALLBACK_MODEL_NAME
        self._model: GenerativeModel = self._build_model()
        logger.info(
            "تم تهيئة GeminiService بنموذج: %s", self._model_name
        )

    # -- بناء النموذج ---------------------------------------------------
    def _build_model(self) -> GenerativeModel:
        """
        يبني كائن GenerativeModel مُهيَّأ بمفتاح API من الإعدادات.

        يُرجع:
            كائن GenerativeModel جاهز للاستدعاء.
        """
        api_key: str = self._settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY غير مُعرَّف في متغيرات البيئة."
            )
        return GenerativeModel(model_name=self._model_name)

    # -- الحصول على محتوى المقال ----------------------------------------
    def get_content(
        self, article_input: ArticleInput
    ) -> str:
        """
        يُحدد ما إذا كان المدخل رابطاً أو نصاً خاماً، ويُرجع المحتوى النصي.

        إذا كان المدخل يحتوي على رابط (url)، يقوم بجلب محتوى المقال من
        الإنترنت. وإلا يُرجع النص المُقدَّم مباشرة.

        المعاملات:
            article_input: كائن ArticleInput يحتوي على الرابط أو النص.

        يُرجع:
            محتوى المقال كسلسلة نصية.
        """
        if article_input.url:
            content: str = fetch_article_content(article_input.url)
            if content:
                return content
            # في حال فشل الجلب، نستخدم النص البديل إن وُجد
            if article_input.text:
                logger.info(
                    "تعذر جلب الرابط، يتم استخدام النص المُقدَّم كبديل."
                )
                return article_input.text
            return ""
        return article_input.text or ""

    # -- استخلاص السيناريو الرئيسي --------------------------------------
    async def extract_script(
        self,
        article_input: ArticleInput,
        max_scenes: int = 8,
    ) -> Script:
        """
        تستقبل رابط مقال أو نصاً خاماً، تقرأ محتواه عبر النموذج
        متعدد الوسائط (gemini-pro-vision / gemini-1.5-pro)، وتستخرج
        سيناريو مقسّماً إلى مشاهد قصيرة (Scenes) مع وصف بصري لكل مشهد.

        تمثّل هذه الدالة المرحلة الأولى الرئيسية من سير العمل.

        المعاملات:
            article_input: كائن ArticleInput يحتوي على الرابط أو النص الخام.
            max_scenes: الحد الأقصى لعدد المشاهد المراد استخراجها (الافتراضي: 8).

        يُرجع:
            كائن Script يحتوي على قائمة المشاهد (scenes) والوصف العام.
        """
        logger.info("بدء مرحلة استخلاص السيناريو (GeminiService.extract_script)")

        # الحصول على المحتوى النصي (من URL أو نص مباشر)
        content: str = self.get_content(article_input)

        if not content or len(content.strip()) < 10:
            logger.warning(
                "المحتوى المُستلم فارغ أو قصير جداً (%d حرف). "
                "يتم إنشاء سيناريو افتراضي.",
                len(content),
            )
            return self._create_fallback_script(article_input)

        # بناء التعليمات (prompt) لاستخلاص المشاهد
        prompt: str = self._build_extraction_prompt(content, max_scenes)

        # استدعاء النموذج مع دعم التنفيذ غير المتزامن
        scenes: List[Scene] = await self._call_gemini_for_scenes(prompt, content)

        # بناء كائن السيناريو النهائي
        script: Script = Script(
            scenes=scenes,
            overall_description=self._generate_overall_description(scenes),
            source_url=article_input.url or "",
            source_text=content[:MAX_CONTENT_CHUNK_LENGTH],
        )

        logger.info(
            "تم استخلاص %d مشهد بنجاح.", len(script.scenes)
        )
        return script

    # -- بناء التعليمات (Prompt) -----------------------------------------
    def _build_extraction_prompt(
        self, content: str, max_scenes: int
    ) -> str:
        """
        يبني التعليمات (prompt) المُرسلة لنموذج Gemini لاستخلاص المشاهد.

        المعاملات:
            content: محتوى المقال النصي.
            max_scenes: الحد الأقصى لعدد المشاهد المطلوبة.

        يُرجع:
            سلسلة التعليمات المُنسّقة بالعربية.
        """
        # تقليم المحتوى إذا تجاوز الحد المسموح
        truncated_content: str = content[:MAX_CONTENT_CHUNK_LENGTH]

        prompt: str = (
            "أنت منتج محتوى متعدد الوسائط خبير. سيتم إعطاؤك مقالاً أو نصاً.\n"
            "مهمتك هي تحليل المحتوى واستخراج سيناريو مقسّم إلى مشاهد قصيرة "
            f"(الحد الأقصى {max_scenes} مشهد) مناسب لإنتاج فيديو عمودي "
            "(Vertical Video) لمنصات Reels / TikTok / Shorts.\n\n"
            "لكل مشهد يجب تقديم:\n"
            "1. عنوان المشهد (بالإنجليزية، مختصر).\n"
            "2. وصف بصري تفصيلي (Visual Description) يُستخدم لتوليد صورة "
            "مناسبة - اشمل التفاصيل التالية: الخلفية، الشخصيات/العناصر، "
            "الإضاءة، الألوان، زاوية الكاميرا.\n"
            "3. النص المراد قراءته في هذا المشهد (Voiceover Text) - بالعربية.\n"
            "4. مدة المشهد التقديرية بالثواني (بين 5 و 15 ثانية).\n"
            "5. ملاحظات بصرية إضافية (مؤثرات، نصوص على الشاشة، انتقالات).\n\n"
            "المحتوى المُقدَّم:\n"
            f"---\n{truncated_content}\n---\n\n"
            "أعد النتيجة بصيغة JSON صالحة تحتوي على مفتاح 'scenes' وهو مصفوفة، "
            "كل عنصر فيه يحتوي على: 'title', 'visual_description', "
            "'voiceover_text', 'duration_seconds', 'visual_notes'.\n"
            "لا تُضف أي نص خارج JSON."
        )
        return prompt

    # -- استدعاء Gemini --------------------------------------------------
    async def _call_gemini_for_scenes(
        self, prompt: str, content: str
    ) -> List[Scene]:
        """
        تستدعي نموذج Gemini لتحليل المحتوى واستخلاص المشاهد.

        تستخدم التنفيذ غير المتزامن (asyncio.to_thread) لتجنب حظر الحلقة
        الحدثية أثناء انتظار استجابة API.

        المعاملات:
            prompt: التعليمات المُنسّقة.
            content: المحتوى النصي المراد تحليله.

        يُرجع:
            قائمة كائنات Scene المستخلصة.
        """
        try:
            scenes: List[Scene] = await asyncio.to_thread(
                self._sync_call_gemini, prompt, content
            )
        except Exception as exc:
            logger.error(
                "حدث خطأ أثناء استدعاء Gemini: %s", exc, exc_info=True
            )
            raise RuntimeError(
                f"فشل استدعاء Gemini API: {exc}"
            ) from exc

        if not scenes:
            logger.warning(
                "لم يتم استخلاص أي مشاهد، يتم إنشاء سيناريو افتراضي."
            )
            return self._create_fallback_scenes(content)

        return scenes

    def _sync_call_gemini(
        self, prompt: str, content: str
    ) -> List[Scene]:
        """
        الاستدعاء المتزامن لنموذج Gemini (يُنفَّذ في خيط منفصل).

        يرسل التعليمات والمحتوى للنموذج، يُحلل الاستجابة، ويُعيد
        قائمة المشاهد.

        المعاملات:
            prompt: التعليمات المُنسّقة.
            content: المحتوى النصي.

        يُرجع:
            قائمة كائنات Scene.
        """
        # محاولة النموذج الأساسي ثم fallback
        model: GenerativeModel = self._model
        response: types.GenerateContentResponse = model.generate_content(
            contents=[prompt, content],
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 8192,
            },
        )

        # محاولة تحليل الاستجابة كـ JSON
        response_text: str = response.text or ""
        scenes: List[Scene] = self._parse_scenes_from_response(response_text)

        if scenes:
            return scenes

        # في حال فشل التحليل، محاولة مع نموذج fallback
        logger.info(
            "محاولة التحليل مع نموذج fallback: %s",
            self._fallback_model_name,
        )
        fallback_model: GenerativeModel = GenerativeModel(
            model_name=self._fallback_model_name
        )
        fallback_response: types.GenerateContentResponse = (
            fallback_model.generate_content(
                contents=[prompt, content],
                generation_config={
                    "temperature": 0.6,
                    "max_output_tokens": 8192,
                },
            )
        )
        fallback_text: str = fallback_response.text or ""
        return self._parse_scenes_from_response(fallback_text)

    # -- تحليل الاستجابة -------------------------------------------------
    def _parse_scenes_from_response(
        self, response_text: str
    ) -> List[Scene]:
        """
        تحلل نص الاستجابة من Gemini لاستخراج المشاهد.

        تحاول استخراج كتلة JSON أولاً، ثم تحللها إلى كائنات Scene.
        في حال فشل JSON، تستخدم تعبيرات منتظمة كحل بديل.

        المعاملات:
            response_text: النص المرتجع من Gemini.

        يُرجع:
            قائمة كائنات Scene المُستخلصة.
        """
        scenes: List[Scene] = []

        # محاولة 1: استخراج JSON من الاستجابة
        json_match: Optional[re.Match[str]] = re.search(
            r"```(?:json)?\s*(\{.*\})\s*```",
            response_text,
            re.DOTALL,
        )
        if json_match:
            try:
                data: Dict[str, Any] = json.loads(json_match.group(1))
                raw_scenes: List[Dict[str, Any]] = data.get("scenes", [])
                scenes = self._build_scenes_from_dicts(raw_scenes)
                if scenes:
                    return scenes
            except json.JSONDecodeError as exc:
                logger.warning("فشل تحليل JSON: %s", exc)

        # محاولة 2: استخراج JSON مباشر بدون markdown
        try:
            # البحث عن أول { حتى آخر } متوازن
            start_idx: int = response_text.find("{")
            if start_idx != -1:
                end_idx: int = self._find_balanced_brace(response_text, start_idx)
                if end_idx != -1:
                    data: Dict[str, Any] = json.loads(
                        response_text[start_idx:end_idx + 1]
                    )
                    raw_scenes: List[Dict[str, Any]] = data.get("scenes", [])
                    scenes = self._build_scenes_from_dicts(raw_scenes)
                    if scenes:
                        return scenes
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("فشل تحليل JSON المباشر: %s", exc)

        # محاولة 3: تحليل نصي باستخدام تعبيرات منتظمة
        scenes = self._parse_scenes_textually(response_text)
        return scenes

    @staticmethod
    def _find_balanced_brace(text: str, start: int) -> int:
        """
        يجد موقع القوس } المتوازن مع القوس { الافتتاحي.

        المعاملات:
            text: النص المراد البحث فيه.
            start: موقع القوس { الافتتاحي.

        يُرجع:
            موقع القوس } المتوازن، أو -1 إن لم يُوجد.
        """
        depth: int = 0
        in_string: bool = False
        escape_next: bool = False
        for i in range(start, len(text)):
            char: str = text[i]
            if escape_next:
                escape_next = False
                continue
            if char == "\\" and in_string:
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def _build_scenes_from_dicts(
        self, raw_scenes: List[Dict[str, Any]]
    ) -> List[Scene]:
        """
        يحوّل قائمة القواميس الخام إلى كائنات Scene.

        المعاملات:
            raw_scenes: قائمة القواميس تحتوي على بيانات المشاهد.

        يُرجع:
            قائمة كائنات Scene.
        """
        scenes: List[Scene] = []
        for index, scene_dict in enumerate(raw_scenes):
            try:
                scene: Scene = Scene(
                    scene_id=index + 1,
                    title=str(scene_dict.get("title", f"Scene {index + 1}")),
                    visual_description=str(
                        scene_dict.get(
                            "visual_description",
                            scene_dict.get("description", ""),
                        )
                    ),
                    voiceover_text=str(
                        scene_dict.get("voiceover_text", "")
                    ),
                    duration_seconds=int(
                        scene_dict.get("duration_seconds", 10)
                    ),
                    visual_notes=str(
                        scene_dict.get("visual_notes", "")
                    ),
                )
                scenes.append(scene)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "تخطي مشهد غير صالح #%d: %s", index + 1, exc
                )
        return scenes

    def _parse_scenes_textually(
        self, response_text: str
    ) -> List[Scene]:
        """
        تحلل المشاهد من النص باستخدام تعبيرات منتظمة كحل بديل.

        تقسّم النص إلى أجزاء وتُنشئ مشاهد من كل قسم.

        المعاملات:
            response_text: النص المرتجع من Gemini.

        يُرجع:
            قائمة كائنات Scene.
        """
        scenes: List[Scene] = []
        # تقسيم النص إلى فقرات
        paragraphs: List[str] = [
            p.strip() for p in response_text.split("\n\n") if p.strip()
        ]

        current_title: str = "Scene 1"
        current_desc: str = ""
        current_voiceover: str = ""
        scene_count: int = 1

        for para in paragraphs:
            if len(para) > 200:
                # فقرة طويلة غالباً وصف بصري أو نص تعليق
                if not current_desc:
                    current_desc = para[:500]
                else:
                    # حفظ المشهد الحالي والبدء بواحد جديد
                    scenes.append(
                        Scene(
                            scene_id=scene_count,
                            title=current_title,
                            visual_description=current_desc,
                            voiceover_text=current_voiceover,
                            duration_seconds=10,
                            visual_notes="",
                        )
                    )
                    scene_count += 1
                    current_title = f"Scene {scene_count}"
                    current_desc = para[:500]
                    current_voiceover = ""
            elif "voiceover" in para.lower() or "تعليق" in para.lower():
                current_voiceover = para[:300]
            elif "scene" in para.lower():
                if current_desc:
                    scenes.append(
                        Scene(
                            scene_id=scene_count,
                            title=current_title,
                            visual_description=current_desc,
                            voiceover_text=current_voiceover,
                            duration_seconds=10,
                            visual_notes="",
                        )
                    )
                    scene_count += 1
                    current_desc = ""
                    current_voiceover = ""
                current_title = para[:100]

        # حفظ آخر مشهد
        if current_desc or current_voiceover:
            scenes.append(
                Scene(
                    scene_id=scene_count,
                    title=current_title,
                    visual_description=current_desc,
                    voiceover_text=current_voiceover,
                    duration_seconds=10,
                    visual_notes="",
                )
            )

        return scenes

    # -- وصف عام ---------------------------------------------------------
    def _generate_overall_description(self, scenes: List[Scene]) -> str:
        """
        يولّد وصفاً عاماً مجمّعاً لجميع المشاهد.

        المعاملات:
            scenes: قائمة المشاهد.

        يُرجع:
            وصف نصي عام مختصر.
        """
        if not scenes:
            return "سيناريو بدون مشاهد."
        titles: List[str] = [scene.title for scene in scenes]
        return " → ".join(titles)

    # -- سيناريو افتراضي (في حال فشل الاستخلاص) --------------------------
    def _create_fallback_script(
        self, article_input: ArticleInput
    ) -> Script:
        """
        يُنشئ سيناريو افتراضي في حال تعذر الاستخلاص.

        المعاملات:
            article_input: كائن ArticleInput الأصلي.

        يُرجع:
            كائن Script افتراضي يحتوي على مشهد واحد.
        """
        fallback_scene: Scene = Scene(
            scene_id=1,
            title="Content Overview",
            visual_description=(
                "مشهد عام يعرض محتوى المقال أو الفكرة المُقدَّمة "
                "من المستخدم."
            ),
            voiceover_text=article_input.text or "محتوى المقال.",
            duration_seconds=15,
            visual_notes="",
        )
        return Script(
            scenes=[fallback_scene],
            overall_description="سيناريو افتراضي (لم يتم استخلاص محتوى).",
            source_url=article_input.url or "",
            source_text=article_input.text or "",
        )

    def _create_fallback_scenes(self, content: str) -> List[Scene]:
        """
        يُنشئ مشاهد افتراضية في حال فشل تحليل الاستجابة.

        المعاملات:
            content: المحتوى النصي الأصلي.

        يُرجع:
            قائمة تحتوي على مشهد افتراضي واحد.
        """
        return [
            Scene(
                scene_id=1,
                title="Content Scene",
                visual_description=content[:300] or "مشهد عام",
                voiceover_text=content[:300] or "",
                duration_seconds=15,
                visual_notes="",
            )
        ]


# ---------------------------------------------------------------------------
# دالة مستقلة للاستدعاء المباشر (module-level convenience function)
# ---------------------------------------------------------------------------
async def extract_script(
    article_input: ArticleInput,
    model_name: Optional[str] = None,
    max_scenes: int = 8,
) -> Script:
    """
    دالة مساعدة مستقلة لاستخلاص السيناريو من مقال أو نص.

    تُنشئ خدمة Gemini داخلياً وتستدعي method extract_script عليها.
    يمكن استخدامها مباشرة دون إنشاء كائن الخدمة يدوياً.

    المعاملات:
        article_input: كائن ArticleInput يحتوي على الرابط أو النص الخام.
        model_name: اسم نموذج Gemini (اختياري).
        max_scenes: الحد الأقصى لعدد المشاهد (الافتراضي: 8).

    يُرجع:
        كائن Script يحتوي على المشاهد المستخلصة.
    """
    service: GeminiService = GeminiService(model_name=model_name)
    return await service.extract_script(
        article_input=article_input, max_scenes=max_scenes
    )
