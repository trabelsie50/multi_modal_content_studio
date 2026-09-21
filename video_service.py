"""
video_service.py — خدمة المونتاج والدمج الآلي للفيديو.
تستخدم MoviePy لدمج الصور المولدة مع التعليق الصوتي وخلفية موسيقية اختيارية
ومؤثرات انتقالية (transitions) لإنتاج فيديو عمودي (1080x1920) متناسق مناسب لـ Reels و TikTok.
"""

import os
import logging
import subprocess
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

from moviepy.editor import (
    ImageClip,
    AudioFileClip,
    CompositeVideoClip,
    CompositeAudioClip,
    concatenate_videoclips,
    concatenate_audioclips,
    VideoFileClip,
    VideoClip,
    TextClip,
)

from models import Script, ImageResult, VoiceoverResult, VideoResult
from config import OUTPUT_DIR, FINAL_VIDEO_DIR, SCENES_DIR

logger = logging.getLogger(__name__)

# ── ثوابت الإخراج ──────────────────────────────────────────────────────────
VERTICAL_WIDTH = 1080
VERTICAL_HEIGHT = 1920
DEFAULT_FPS = 30
DEFAULT_TRANSITION_DURATION = 0.5
MIN_SCENE_DURATION = 2.0
MAX_SCENE_DURATION = 12.0
BACKGROUND_MUSIC_VOLUME = 0.15
VOICEOVER_VOLUME = 1.0
DEFAULT_VIDEO_BITRATE = "4000k"
DEFAULT_AUDIO_BITRATE = "192k"
DEFAULT_VIDEO_FILENAME = "final_output.mp4"
ARABIC_WORDS_PER_MINUTE = 120
ENGLISH_WORDS_PER_MINUTE = 150
conservative_wpm = 130


class VideoService:
    """خدمة المونتاج والدمج الآلي للفيديو باستخدام MoviePy.

    تجمع الصور المولدة لكل مشهد مع التعليق الصوتي وخلفية موسيقية اختيارية
    وإضافات انتقالية لإنتاج فيديو عمودي (1080x1920) جاهز للنشر.
    """

    def __init__(
        self,
        output_dir: str = FINAL_VIDEO_DIR,
        background_music: Optional[str] = None,
    ):
        """تهيئة خدمة المونتاج.

        Args:
            output_dir: المسار الذي سيُحفظ فيه الفيديو النهائي.
            background_music: مسار ملف الموسيقى الخلفية (اختياري).
        """
        self.output_dir = output_dir
        self.background_music_path = background_music
        self.fps = DEFAULT_FPS
        self.video_codec = "libx264"
        self.audio_codec = "aac"
        self.encoding_preset = "medium"
        self.video_bitrate = DEFAULT_VIDEO_BITRATE
        self.audio_bitrate = DEFAULT_AUDIO_BITRATE
        self.encoding_threads = 4
        self.transition_duration = DEFAULT_TRANSITION_DURATION
        self.default_scene_duration = MIN_SCENE_DURATION
        self.text_font_size = 24
        self.text_color = "white"
        self.text_font = "Arial"
        self.video_format = "mp4"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(SCENES_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── الدالة الرئيسية: تجميع الفيديو ────────────────────────────────────
    def assemble_video(
        self,
        images: List[ImageResult],
        voiceover: VoiceoverResult,
        script: Script,
        transitions: bool = True,
        background_music_path: Optional[str] = None,
    ) -> VideoResult:
        """تجميع الفيديو النهائي من الصور والتعليق الصوتي والسيناريو.

        مرحلة الدمج النهائية في سير العمل — تستقبل نتائج جميع المراحل السابقة
        (توليد الصور، توليد الصوت) وتنتج فيديو عمودي متناسق.

        Args:
            images: قائمة نتائج الصور المولدة لكل مشهد.
            voiceover: نتيجة التعليق الصوتي (الملف الصوتي ومدته).
            script: السيناريو المستخرج من المقال.
            transitions: تفعيل المؤثرات الانتقالية بين المشاهد.
            background_music_path: مسار الموسيقى الخلفية (يُغطي القيمة الافتراضية).

        Returns:
            VideoResult يحتوي مسار الفيديو ومدته.

        Raises:
            ValueError: إذا لم تكن هناك صور للتجميع.
            FileNotFoundError: إذا لم يكن ملف صورة موجوداً.
        """
        if not images:
            raise ValueError("لا توجد صور لتجميع الفيديو — تأكد من توليد الصور أولاً.")

        # التحقق من وجود ملفات الصور
        for img in images:
            if not os.path.isfile(img.file_path):
                raise FileNotFoundError(
                    f"ملف الصورة غير موجود: {img.file_path} — تأكد من توليدها بنجاح."
                )

        logger.info(
            "بدء تجميع الفيديو: %d مشهد، %d صورة، انتقالات=%s",
            len(images),
            len(images),
            transitions,
        )

        # ── الخطوة 1: حساب المدة الإجمالية للفيديو ──────────────
        total_duration = self._get_total_duration(voiceover, script)
        logger.info("المدة الإجمالية المحسوبة: %.1f ثانية", total_duration)

        # ── الخطوة 2: توزيع المدة على كل مشهد ────────────────────────────
        scene_durations = self._calculate_scene_durations(len(images), total_duration)
        for i, dur in enumerate(scene_durations):
            logger.info("  مشهد %d: %.1f ثانية", i + 1, dur)

        # ── الخطوة 3: إنشاء مقاطع الصور ──────────────────────────
        image_clips: List[ImageClip] = []
        for idx, (img, duration) in enumerate(zip(images, scene_durations)):
            clip = self._create_image_clip(img, duration)
            image_clips.append(clip)

        # ── الخطوة 4: تطبيق الانتقالات بين المشاهد ─────────────
        if transitions and len(image_clips) > 1:
            default_transition_types = ["fadein", "fadeout"] * (len(image_clips) // 2 + 1)
            image_clips = self._apply_transitions(image_clips, default_transition_types)

        # ── الخطوة 5: دمج المقاطع ─────────────────────
        final_clip = self._concatenate_clips(image_clips)

        # ── الخطوة 6: إضافة التعليق الصوتي ────────────────────
        if voiceover:
            final_clip = self._add_voiceover(final_clip, voiceover)

        # ── الخطوة 7: إضافة النصوص من السكريبت ─────────────────
        if script:
            final_clip = self._add_script_overlay(final_clip, script)

        # ── الخطوة 8: تصدير الفيديو النهائي ───────────────────
        output_path = self._render_video(final_clip, self.video_format)

        # ── الخطوة 9: حساب مدة الفيديو النهائي ─────────────────
        final_duration = self._get_final_duration(output_path)

        logger.info("تم إنشاء الفيديو بنجاح: %s، المدة: %.1f ثانية", output_path, final_duration)

        return VideoResult(file_path=output_path, duration=final_duration)

    # ════════════════════════════════════════════════════════════
    #  طرق مساعدة لتجميع الفيديو
    # ════════════════════════════════════════════════════════════

    def _create_image_clip(self, image: ImageResult, duration: float) -> ImageClip:
        """إنشاء مقطع صورة من ملف صورة مع مدة محددة.

        Args:
            image: كائن ImageResult يحتوي مسار الصورة.
            duration: مدة المقطع بالثواني.

        Returns:
            ImageClip جاهز للدمج.
        """
        clip = ImageClip(image.file_path).set_duration(duration)
        logger.debug("تم إنشاء مقطع صورة: %s، المدة: %.1f ثانية", image.file_path, duration)
        return clip

    def _apply_transitions(
        self, clips: List[ImageClip], transition_types: List[str]
    ) -> List[ImageClip]:
        """تطبيق تأثيرات الانتقال بين مقاطع الصور.

        Args:
            clips: قائمة مقاطع الصور.
            transition_types: قائمة أنواع الانتقالات.

        Returns:
            قائمة المقاطع بعد تطبيق الانتقالات.
        """
        if len(clips) <= 1:
            return clips

        transitioned_clips: List[ImageClip] = []
        num_transitions = min(len(transition_types), len(clips) - 1)

        for i, clip in enumerate(clips):
            if i < num_transitions:
                transition_type = transition_types[i]
                logger.info("تطبيق انتقال '%s' على المشهد %d", transition_type, i + 1)
                clip = self._apply_single_transition(clip, transition_type)
            transitioned_clips.append(clip)

        return transitioned_clips

    def _apply_single_transition(self, clip: ImageClip, transition_type: str) -> ImageClip:
        """تطبيق تأثير انتقال واحد على مقطع.

        Args:
            clip: مقطع الصورة الأصلي.
            transition_type: نوع الانتقال (fadein, fadeout, slide, zoom).

        Returns:
            مقطع الصورة بعد تطبيق التأثير.
        """
        duration = clip.duration or 1.0
        transition_duration = min(self.transition_duration, duration / 2)

        if transition_type == "fadein":
            clip = clip.crossfadein(transition_duration)
        elif transition_type == "fadeout":
            clip = clip.crossfadeout(transition_duration)
        elif transition_type == "slide":
            clip = clip.resize(width=int(clip.w * 1.1)).margin(
                x1=0, y1=0, x2=int(clip.w * 0.05), y2=0
            ).set_duration(duration)
        elif transition_type == "zoom":
            clip = clip.resize(lambda t: 1.0 + 0.1 * (1 - t / duration)).set_duration(duration)
        else:
            logger.warning("نوع الانتقال غير معروف: %s — يتم تخطيه", transition_type)

        return clip

    def _concatenate_clips(self, clips: List[ImageClip]) -> CompositeVideoClip:
        """دمج قائمة المقاطع في مقطع واحد متصل.

        Args:
            clips: قائمة المقاطع المراد دمجها.

        Returns:
            CompositeVideoClip يحتوي جميع المقاطع المدمجة.
        """
        if not clips:
            raise ValueError("لا توجد مقاطع للدمج.")
        if len(clips) == 1:
            return clips[0]

        concatenated = concatenate_videoclips(clips, method="compose", padding=-1)
        logger.info("تم دمج %d مقطع في مقطع واحد", len(clips))
        return concatenated

    def _add_voiceover(
        self, clip: VideoClip, voiceover: VoiceoverResult
    ) -> VideoClip:
        """إضافة التعليق الصوتي إلى مقطع الفيديو.

        Args:
            clip: مقطع الفيديو الأساسي.
            voiceover: كائن VoiceoverResult يحتوي مسار ملف الصوت.

        Returns:
            VideoClip مع التعليق الصوتي المدمج.
        """
        if not voiceover.audio_path or not os.path.isfile(voiceover.audio_path):
            logger.warning("ملف التعليق الصوتي غير موجود: %s — يتم التخطي", voiceover.audio_path)
            return clip

        audio = AudioFileClip(voiceover.audio_path)

        if audio.duration > clip.duration:
            audio = audio.subclip(0, clip.duration)
        else:
            audio = audio.set_duration(clip.duration)

        clip = clip.set_audio(audio)
        logger.info("تم إضافة التعليق الصوتي: %s", voiceover.audio_path)
        return clip

    def _add_script_overlay(
        self, clip: VideoClip, script: Script
    ) -> VideoClip:
        """إضافة نصوص السكريبت كتراكب على مقطع الفيديو.

        Args:
            clip: مقطع الفيديو الأساسي.
            script: كائن Script يحتوي نصوص المشاهد.

        Returns:
            VideoClip مع النصوص المضافة.
        """
        if not script.text:
            logger.warning("نص السكريبت فارغ — يتم التخطي")
            return clip

        txt_clip = TextClip(
            script.text,
            fontsize=self.text_font_size,
            color=self.text_color,
            font=self.text_font,
            size=(clip.w - 40, None),
            method="caption",
        )
        txt_duration = clip.duration or 1.0
        txt_clip = txt_clip.set_duration(txt_duration)

        txt_position = (
            int(clip.w / 2 - txt_clip.w / 2),
            int(clip.h - txt_clip.h - 30),
        )
        txt_clip = txt_clip.set_position(txt_position)

        clip = CompositeVideoClip([clip, txt_clip])
        logger.info("تم إضافة نص السكريبت كتراكب")
        return clip

    def _render_video(self, clip: VideoClip, fmt: str = "mp4") -> str:
        """تصدير مقطع الفيديو إلى ملف.

        Args:
            clip: مقطع الفيديو المراد تصديره.
            fmt: صيغة التصدير (mp4, webm, avi).

        Returns:
            مسار الملف المصدَّر.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"video_{timestamp}.{fmt}")

        logger.info("بدء تصدير الفيديو إلى: %s", output_path)
        clip.write_videofile(
            output_path,
            fps=self.fps,
            codec=self.video_codec,
            audio_codec=self.audio_codec,
            preset=self.encoding_preset,
            bitrate=self.video_bitrate,
            threads=self.encoding_threads,
            logger="bar",
        )
        logger.info("اكتمل تصدير الفيديو: %s", output_path)
        return output_path

    def _get_final_duration(self, video_path: str) -> float:
        """حساب مدة ملف الفيديو.

        Args:
            video_path: مسار ملف الفيديو.

        Returns:
            مدة الفيديو بالثواني.
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                capture_output=True,
                text=True,
            )
            duration = float(result.stdout.strip())
            logger.info("مدة الفيديو النهائي: %.1f ثانية", duration)
            return duration
        except (ValueError, IndexError, FileNotFoundError) as exc:
            logger.warning("تعذر قراءة مدة الفيديو عبر ffprobe: %s", exc)
            return 0.0

    # ════════════════════════════════════════════════════════════
    #  طرق مساعدة لحساب المدد
    # ════════════════════════════════════════════════════════════

    def _get_total_duration(
        self, voiceover: Optional[VoiceoverResult] = None,
        script: Optional[Script] = None,
    ) -> float:
        """حساب المدة الإجمالية للفيديو من التعليق الصوتي أو السكريبت.

        Args:
            voiceover: كائن التعليق الصوتي (اختياري).
            script: كائن السكريبت (اختياري).

        Returns:
            المدة الإجمالية بالثواني.
        """
        duration = self.default_scene_duration

        if voiceover and voiceover.audio_path and os.path.isfile(voiceover.audio_path):
            try:
                audio = AudioFileClip(voiceover.audio_path)
                duration = max(duration, audio.duration)
                audio.close()
            except Exception as exc:
                logger.warning("تعذر قراءة مدة التعليق الصوتي: %s", exc)

        if script and script.duration:
            duration = max(duration, script.duration)

        logger.info("المدة الإجمالية المحسوبة من المصادر: %.1f ثانية", duration)
        return duration

    def _calculate_scene_durations(self, num_scenes: int, total_duration: float) -> List[float]:
        """توزيع المدة الإجمالية بالتساوي على جميع المشاهد.

        Args:
            num_scenes: عدد المشاهد.
            total_duration: المدة الإجمالية بالثواني.

        Returns:
            قائمة مدد كل مشهد.
        """
        if num_scenes <= 0:
            return []

        per_scene = total_duration / num_scenes
        durations = [round(per_scene, 2) for _ in range(num_scenes)]

        diff = round(total_duration - sum(durations), 2)
        if diff != 0:
            durations[-1] = round(durations[-1] + diff, 2)

        return durations
