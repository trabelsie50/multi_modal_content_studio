"""
Service for generating and enhancing images for each video scene.
Uses Google Imagen API (via Vertex AI) to generate expressive images from scene descriptions,
with a quality enhancement (upscale) function that re-invokes the model with detailed
improvements to ensure high resolution suitable for vertical video production.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from google.cloud import aiplatform
from google.oauth2 import service_account

from config import IMAGEN_API_KEY, OUTPUT_DIR, SCENES_DIR
from models import ImageResult, Scene

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level convenience constants
# ---------------------------------------------------------------------------
DEFAULT_IMAGEN_MODEL = "imagen-3.0"
DEFAULT_LOCATION = "us-central1"
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920
DEFAULT_ASPECT_RATIO = "9:16"
DEFAULT_NUMBER_OF_IMAGES = 1
REQUEST_TIMEOUT = 120  # seconds for HTTP requests


class ImageService:
    """
    Service for generating and enhancing images for video scenes using
    Google Imagen API via Vertex AI.

    Workflow:
        1. generate_scene_image  – creates an image from a Scene description.
        2. enhance_image         – upscales / improves the generated image
           by re-invoking Imagen with detailed quality prompts.
    """

    def __init__(
        self,
        api_key: str = IMAGEN_API_KEY,
        project_id: Optional[str] = None,
        location: str = DEFAULT_LOCATION,
        output_dir: str = SCENES_DIR,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the ImageService.

        Args:
            api_key:         Imagen API key (from environment or explicit).
            project_id:      Google Cloud project ID.
            location:        Vertex AI region (default us-central1).
            output_dir:      Directory to store generated images.
            model_name:      Imagen model identifier (default imagen-3.0).
        """
        self.api_key: str = api_key
        self.project_id: str = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "")
        self.location: str = location
        self.model_name: str = model_name or DEFAULT_IMAGEN_MODEL
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cached Vertex AI initialisation
        self._initialized: bool = False
        self._model: Optional[Any] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _initialize(self) -> None:
        """Initialize the Vertex AI client and load the Imagen model."""
        if self._initialized and self._model is not None:
            return

        credentials = None
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and os.path.exists(creds_path):
            credentials = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            logger.info("Loaded service account credentials from: %s", creds_path)

        aiplatform.init(
            project=self.project_id,
            location=self.location,
            credentials=credentials,
        )
        self._model = aiplatform.ImageGenerationModel.from_pretrained(self.model_name)
        self._initialized = True
        logger.info(
            "Vertex AI initialised – model=%s location=%s project=%s",
            self.model_name,
            self.location,
            self.project_id,
        )

    # ------------------------------------------------------------------
    # Public API – image generation
    # ------------------------------------------------------------------
    async def generate_scene_image(
        self,
        scene: Scene,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        number_of_images: int = DEFAULT_NUMBER_OF_IMAGES,
    ) -> ImageResult:
        """
        Generate an expressive image for a single scene using Google Imagen.

        Args:
            scene:            The scene dataclass containing description/prompt.
            aspect_ratio:     Target aspect ratio (default 9:16 for vertical video).
            number_of_images: How many variants to generate (default 1).

        Returns:
            ImageResult dataclass with path, URL, prompt, and metadata.
        """
        self._initialize()

        prompt = self._build_generation_prompt(scene)
        scene_dir = self._get_scene_directory(scene.scene_id)
        scene_dir.mkdir(parents=True, exist_ok=True)

        original_path = scene_dir / f"scene_{scene.scene_id}_original.png"

        logger.info(
            "Generating image for scene %s | prompt: %.120s",
            scene.scene_id,
            prompt,
        )

        start_time = time.time()
        response = await self._async_call_imagen(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            number_of_images=number_of_images,
        )
        generation_duration = time.time() - start_time

        image_uri = self._extract_image_uri(response)
        image_bytes = await self._async_download_image(image_uri)

        with open(original_path, "wb") as f:
            f.write(image_bytes)

        file_size = original_path.stat().st_size

        result = ImageResult(
            scene_id=scene.scene_id,
            image_path=str(original_path),
            image_url=image_uri,
            prompt_used=prompt,
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            quality="original",
            generation_duration=generation_duration,
            file_size=file_size,
        )

        logger.info(
            "Image generated for scene %s → %s (%.2f s, %d bytes)",
            scene.scene_id,
            original_path,
            generation_duration,
            file_size,
        )
        return result

    # ------------------------------------------------------------------
    # Public API – image enhancement / upscaling
    # ------------------------------------------------------------------
    def enhance_image(
        self,
        image_result: ImageResult,
        scene: Scene,
        target_width: int = DEFAULT_WIDTH,
        target_height: int = DEFAULT_HEIGHT,
    ) -> ImageResult:
        """
        Enhance / upscale a previously generated image by re-invoking Imagen
        with a detailed quality-improvement prompt.

        The method takes the original prompt, adds explicit quality directives
        (8K, HDR, sharp detail, no artifacts), and asks Imagen to produce a
        higher-fidelity version suitable for vertical video output.

        Args:
            image_result:   The original ImageResult to enhance.
            scene:          The original Scene for context.
            target_width:   Desired output width in pixels.
            target_height:  Desired output height in pixels.

        Returns:
            A new ImageResult pointing to the enhanced image file.
        """
        self._initialize()

        enhanced_prompt = self._build_enhancement_prompt(
            original_prompt=image_result.prompt_used,
            scene=scene,
            target_width=target_width,
            target_height=target_height,
        )

        scene_dir = self._get_scene_directory(scene.scene_id)
        scene_dir.mkdir(parents=True, exist_ok=True)
        enhanced_path = scene_dir / f"scene_{scene.scene_id}_enhanced.png"

        logger.info(
            "Enhancing image for scene %s with detailed prompt | %.120s",
            scene.scene_id,
            enhanced_prompt,
        )

        start_time = time.time()
        response = self._call_imagen_sync(
            prompt=enhanced_prompt,
            aspect_ratio=DEFAULT_ASPECT_RATIO,
            number_of_images=DEFAULT_NUMBER_OF_IMAGES,
        )
        generation_duration = time.time() - start_time

        image_uri = self._extract_image_uri(response)
        image_bytes = self._download_image_sync(image_uri)

        with open(enhanced_path, "wb") as f:
            f.write(image_bytes)

        file_size = enhanced_path.stat().st_size

        enhanced_result = ImageResult(
            scene_id=scene.scene_id,
            image_path=str(enhanced_path),
            image_url=image_uri,
            prompt_used=enhanced_prompt,
            width=target_width,
            height=target_height,
            quality="enhanced",
            generation_duration=generation_duration,
            file_size=file_size,
        )

        logger.info(
            "Image enhanced for scene %s → %s (%.2f s, %d bytes)",
            scene.scene_id,
            enhanced_path,
            generation_duration,
            file_size,
        )
        return enhanced_result

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------
    @staticmethod
    def _build_generation_prompt(scene: Scene) -> str:
        """
        Build a detailed Imagen prompt from a Scene's description and visual notes.
        """
        # Prefer explicit visual_prompt if available, fall back to description
        scene_desc = ""
        if hasattr(scene, "visual_prompt") and scene.visual_prompt:
            scene_desc = scene.visual_prompt
        elif hasattr(scene, "description") and scene.description:
            scene_desc = scene.description
        else:
            scene_desc = str(getattr(scene, "scene_id", "unknown"))

        prompt_parts: List[str] = [
            "Professional cinematic photograph, vertical format (9:16 aspect ratio),",
            "high resolution, vibrant saturated colors, dramatic studio lighting,",
            "sharp focus, clean composition, suitable for social media video content.",
            f"Scene description: {scene_desc}",
        ]
        return " ".join(prompt_parts)

    @staticmethod
    def _build_enhancement_prompt(
        original_prompt: str,
        scene: Scene,
        target_width: int,
        target_height: int,
    ) -> str:
        """
        Build an enhanced prompt that adds explicit quality directives for
        upscaling / improving image fidelity.
        """
        scene_desc = ""
        if hasattr(scene, "description") and scene.description:
            scene_desc = scene.description

        prompt_parts: List[str] = [
            "ULTRA HIGH QUALITY cinematic photograph, 8K resolution,",
            f"{target_width}x{target_height} pixels, extremely sharp details,",
            "professional color grading, HDR, wide dynamic range,",
            "studio lighting with soft shadows, photorealistic rendering,",
            "no compression artifacts, no blur, no noise, no distortion,",
            "clean edges, premium stock photo quality,",
            "suitable for vertical video production (Reels / TikTok / Shorts).",
            f"Original concept: {original_prompt}",
        ]
        if scene_desc:
            prompt_parts.append(f"Scene context: {scene_desc}")

        return " ".join(prompt_parts)

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------
    def _get_scene_directory(self, scene_id: int) -> Path:
        """Return the directory for a specific scene's images."""
        return self.output_dir / f"scene_{scene_id}"

    # ------------------------------------------------------------------
    # Imagen API calls (sync + async wrappers)
    # ------------------------------------------------------------------
    def _call_imagen_sync(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        number_of_images: int = DEFAULT_NUMBER_OF_IMAGES,
    ) -> Any:
        """
        Synchronous call to Google Imagen via Vertex AI ImageGenerationModel.

        Returns the raw response object from generate_images().
        """
        if self._model is None:
            raise RuntimeError("ImageService not initialised. Call _initialize() first.")

        response = self._model.generate_images(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            number_of_images=number_of_images,
        )
        return response

    async def _async_call_imagen(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        number_of_images: int = DEFAULT_NUMBER_OF_IMAGES,
    ) -> Any:
        """
        Asynchronous wrapper around the synchronous Imagen call.
        Runs the blocking Vertex AI call in a thread executor so that
        the event loop remains responsive.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._call_imagen_sync(
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                number_of_images=number_of_images,
            ),
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_image_uri(response: Any) -> str:
        """
        Extract the first image URI from an Imagen API response.

        The Vertex AI ImageGenerationModel response typically exposes a list
        of generated images, each with a `uri` attribute pointing to a
        Google Cloud Storage object.
        """
        images: List[Any] = []

        if hasattr(response, "images") and response.images:
            images = response.images
        elif hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "images") and candidate.images:
                    images = candidate.images
                    break

        if not images:
            raise RuntimeError(
                "Imagen API returned no images in the response."
            )

        first_image = images[0]

        if hasattr(first_image, "uri") and first_image.uri:
            return first_image.uri

        # Some SDK versions return the image as bytes directly
        if hasattr(first_image, "bytes") and first_image.bytes:
            # Save bytes to a temporary location and return a local path
            return "bytes"

        raise RuntimeError(
            "Could not extract image URI from Imagen response object."
        )

    # ------------------------------------------------------------------
    # Image download helpers
    # ------------------------------------------------------------------
    def _download_image_sync(self, uri: str) -> bytes:
        """
        Synchronously download image bytes from a URI (GCS signed URL or public URL).
        """
        if uri == "bytes":
            raise RuntimeError(
                "Direct bytes download not implemented for this Imagen response format."
            )

        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.info("Downloading image from: %.100s", uri)
        response = requests.get(uri, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.content

    async def _async_download_image(self, uri: str) -> bytes:
        """
        Asynchronously download image bytes by offloading to a thread.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._download_image_sync(uri),
        )


# ======================================================================
# Module-level convenience functions (used by orchestrator)
# ======================================================================

async def generate_scene_image(
    scene: Scene,
    service: Optional[ImageService] = None,
) -> ImageResult:
    """
    Convenience async function to generate an image for a scene.

    If no ImageService instance is provided, a default one is created
    using environment variables.

    Args:
        scene:   The Scene dataclass.
        service: Optional ImageService instance.

    Returns:
        ImageResult with the generated image metadata.
    """
    if service is None:
        service = ImageService()
    return await service.generate_scene_image(scene)


def enhance_image(
    image_result: ImageResult,
    scene: Scene,
    target_width: int = DEFAULT_WIDTH,
    target_height: int = DEFAULT_HEIGHT,
) -> ImageResult:
    """
    Convenience function to enhance / upscale a generated image.

    Creates a default ImageService if none is supplied.

    Args:
        image_result:   The original ImageResult to enhance.
        scene:          The Scene dataclass for context.
        target_width:   Output width in pixels.
        target_height:  Output height in pixels.

    Returns:
        ImageResult pointing to the enhanced image file.
    """
    service = ImageService()
    return service.enhance_image(
        image_result=image_result,
        scene=scene,
        target_width=target_width,
        target_height=target_height,
    )
