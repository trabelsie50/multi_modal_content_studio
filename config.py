import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    gemini_api_key: str
    tts_api_key: str
    elevenlabs_api_key: str
    google_application_credentials: str
    imagen_api_key: str
    output_dir: str
    scenes_dir: str
    final_video_dir: str


def get_settings() -> Settings:
    output_dir = os.getenv("OUTPUT_DIR", "output")
    scenes_dir = os.path.join(output_dir, "scenes")
    final_video_dir = os.path.join(output_dir, "final_videos")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(scenes_dir):
        os.makedirs(scenes_dir, exist_ok=True)
    if not os.path.exists(final_video_dir):
        os.makedirs(final_video_dir, exist_ok=True)

    return Settings(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        tts_api_key=os.getenv("TTS_API_KEY", ""),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        google_application_credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
        imagen_api_key=os.getenv("IMAGEN_API_KEY", ""),
        output_dir=output_dir,
        scenes_dir=scenes_dir,
        final_video_dir=final_video_dir,
    )


_SETTINGS = get_settings()

TELEGRAM_BOT_TOKEN = _SETTINGS.telegram_bot_token
GEMINI_API_KEY = _SETTINGS.gemini_api_key
TTS_API_KEY = _SETTINGS.tts_api_key
ELEVENLABS_API_KEY = _SETTINGS.elevenlabs_api_key
GOOGLE_APPLICATION_CREDENTIALS = _SETTINGS.google_application_credentials
IMAGEN_API_KEY = _SETTINGS.imagen_api_key
OUTPUT_DIR = _SETTINGS.output_dir
SCENES_DIR = _SETTINGS.scenes_dir
FINAL_VIDEO_DIR = _SETTINGS.final_video_dir
