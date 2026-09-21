import logging
import os

from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from config import TELEGRAM_BOT_TOKEN, get_settings
from orchestrator import run_pipeline
from models import ArticleInput

logger = logging.getLogger(__name__)


async def post_init_callback(application: Application) -> None:
    """Called after the Application is initialized.

    Sets the bot's command list and performs any startup tasks.
    """
    settings = get_settings()
    logger.info(
        "Multi-Modal Content Studio bot initialized. "
        "Output dir: %s, Scenes dir: %s, Final video dir: %s",
        settings.OUTPUT_DIR,
        settings.SCENES_DIR,
        settings.FINAL_VIDEO_DIR,
    )
    await application.bot.set_my_commands(
        [
            BotCommand("start", "Start the bot and begin content creation."),
        ]
    )
    logger.info("Bot commands registered successfully.")


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command.

    Sends a welcome message explaining the bot's capabilities.
    """
    user = update.effective_user
    if user is None or update.message is None:
        return

    welcome_text = (
        f"مرحباً {user.first_name}! 👋\n\n"
        "أنا بوت استوديو المحتوى التلقائي متعدد الوسائط 🎬\n\n"
        "أرسل لي رابط مقال أو فكرة نصية عشوائية، وسأقوم بـ:\n\n"
        "📝 تحليل المحتوى واستخراج سيناريو مقسّم لمشاهد\n"
        "🖼️ توليد صور معبرة لكل مشهد\n"
        "🔊 إنشاء تعليق صوتي احترافي\n"
        "🎞️ إنتاج فيديو عمودي جاهز للنشر (Reels / TikTok)\n\n"
        "ابدأ الآن بإرسال رابط مقال أو فكرتك! 🚀"
    )
    await update.message.reply_text(welcome_text)


async def handle_content_request(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handles text messages containing URLs or text ideas.

    Receives the user's input, creates an ArticleInput, invokes the
    Orchestrator (run_pipeline) to process the content through all stages,
    and sends back the final video and script.
    """
    user = update.effective_user
    message = update.message

    if user is None or message is None:
        return

    text = message.text.strip() if message.text else ""

    if not text:
        await message.reply_text("يرجى إرسال رابط مقال أو فكرة نصية.")
        return

    # Determine if input is a URL or plain text
    is_url = text.startswith(("http://", "https://"))

    # Send initial processing message
    processing_msg = await message.reply_text(
        f"🔄 جاري معالجة طلبك {user.first_name}...\n"
        "📝 المرحلة الأولى: تحليل المحتوى واستخراج السيناريو..."
    )

    try:
        # Create ArticleInput dataclass from user message
        article_input = ArticleInput(
            url=text if is_url else None,
            text=text if not is_url else None,
            title=(
                f"محتوى من {user.first_name}"
                if not is_url
                else f"مقال: {text[:100]}"
            ),
        )

        # Stage 1: Extract script via Gemini (orchestrated by run_pipeline)
        await processing_msg.edit_text(
            f"🔄 جاري معالجة طلبك {user.first_name}...\n"
            "📝 المرحلة الأولى: تحليل المحتوى واستخراج السيناريو...\n"
            "⏳ يرجى الانتظار..."
        )

        # Run the full pipeline (all stages orchestrated)
        pipeline_result = await run_pipeline(article_input)

        # Stage 2-4 handled inside run_pipeline:
        # - Image generation for each scene
        # - Voiceover generation via ElevenLabs
        # - Video assembly via MoviePy

        await processing_msg.edit_text(
            f"🔄 جاري معالجة طلبك {user.first_name}...\n"
            "🎬 المرحلة النهائية: إنتاج الفيديو...\n"
            "⏳ يرجى الانتظار..."
        )

        # --- Deliver results ---

        # 1. Send the final video if available
        video_result = pipeline_result.video_result
        if video_result is not None and video_result.file_path:
            try:
                video_file = open(video_result.file_path, "rb")
                await message.reply_video(
                    video=video_file,
                    caption=(
                        "✅ هذا هو الفيديو النهائي الجاهز للنشر!\n"
                        "شكراً لاستخدامك استوديو المحتوى التلقائي 🎉"
                    ),
                    supports_streaming=True,
                )
                video_file.close()
            except (FileNotFoundError, OSError) as file_err:
                logger.error(
                    "Could not send video file: %s", str(file_err), exc_info=True
                )
                await message.reply_text(
                    "⚠️ تم إنتاج الفيديو لكن حدث مشكلة في إرساله. "
                    "يرجى المحاولة مرة أخرى."
                )
        else:
            await message.reply_text(
                "⚠️ لم يتم إنتاج الفيديو. يرجى مراجعة المدخلات والمحاولة مرة أخرى."
            )

        # 2. Send the script text if available
        script = pipeline_result.script
        if script is not None:
            script_lines = [
                "📝 سيناريو المحتوى المُنتج:",
                "",
            ]

            if script.title:
                script_lines.append(f"العنوان: {script.title}")
                script_lines.append("")

            if script.scenes:
                for scene in script.scenes:
                    script_lines.append(
                        f"المشهد {scene.scene_id}: {scene.description}"
                    )
                    if scene.scene_text:
                        script_lines.append(f"النص: {scene.scene_text}")
                    if scene.duration_seconds:
                        script_lines.append(
                            f"المدة: {scene.duration_seconds} ثانية"
                        )
                    script_lines.append("")

            if script.notes:
                script_lines.append(f"ملاحظات: {script.notes}")
                script_lines.append("")

            script_text = "\n".join(script_lines)

            # Telegram message limit is 4096 characters; split if needed
            max_len = 4000
            if len(script_text) <= max_len:
                await message.reply_text(script_text)
            else:
                for i in range(0, len(script_text), max_len):
                    chunk = script_text[i : i + max_len]
                    await message.reply_text(chunk)
        else:
            await message.reply_text(
                "⚠️ لم يتم إنشاء سيناريو نصي."
            )

        # 3. Send image results if available
        if pipeline_result.images:
            for idx, img_result in enumerate(pipeline_result.images):
                if img_result.file_path and os.path.exists(img_result.file_path):
                    try:
                        caption = (
                            f"🖼️ صورة المشهد {img_result.scene_id}"
                            if img_result.scene_id
                            else f"🖼️ صورة {idx + 1}"
                        )
                        img_file = open(img_result.file_path, "rb")
                        await message.reply_photo(
                            photo=img_file,
                            caption=caption,
                        )
                        img_file.close()
                    except (FileNotFoundError, OSError) as img_err:
                        logger.error(
                            "Could not send image file: %s", str(img_err)
                        )

        # Mark processing as complete
        await processing_msg.edit_text(
            f"✅ تمت معالجة طلب {user.first_name} بنجاح! 🎉\n"
            f"تم إنتاج {len(pipeline_result.images) if pipeline_result.images else 0} صورة "
            f"و{len(script.scenes) if script and script.scenes else 0} مشهد."
        )

    except Exception as e:
        logger.error(
            "Error processing content request from user %s: %s",
            user.id if user else "unknown",
            str(e),
            exc_info=True,
        )
        try:
            await processing_msg.edit_text(
                f"❌ حدث خطأ أثناء المعالجة:\n\n"
                f"{type(e).__name__}: {str(e)}\n\n"
                f"يرجى التأكد من صحة الرابط أو المحاولة مرة أخرى."
            )
        except Exception:
            # If we can't edit the message (e.g., it was deleted), just log
            pass


def main() -> None:
    """Entry point for the Multi-Modal Content Studio Telegram bot.

    Creates the async Application, registers handlers, and starts polling.
    """
    settings = get_settings()

    # Build the async Application with the bot token and post-init callback
    application = (
        Application.builder()
        .token(settings.TELEGRAM_BOT_TOKEN)
        .post_init(post_init_callback)
        .build()
    )

    # Register command handler for /start
    application.add_handler(CommandHandler("start", handle_start))

    # Register text handler for URLs and plain text ideas
    # Filters out commands (messages starting with /) to avoid conflicts
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_content_request,
        )
    )

    # Start the bot with polling
    logger.info(
        "Starting Multi-Modal Content Studio bot on %s",
        settings.TELEGRAM_BOT_TOKEN[:10] + "...",
    )
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
