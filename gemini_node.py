import sys
import os
import warnings

# Suppress warnings related to IMAGE_SAFETY finish reason which is normal
warnings.filterwarnings("ignore", message="IMAGE_SAFETY is not a valid FinishReason")

# Add site-packages directory to Python's sys.path
'''
site_packages_path = os.path.join(sys.prefix, 'Lib', 'site-packages')
if site_packages_path not in sys.path:
    sys.path.insert(0, site_packages_path)
'''
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import logging
import random
import base64
import time
import uuid
import hashlib
import requests
import json
import shutil
import subprocess
import tempfile

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


SUPPORTED_VIDEO_EXTS = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".flv": "video/x-flv",
    ".wmv": "video/x-ms-wmv",
    ".3gp": "video/3gpp",
    ".mkv": "video/x-matroska",
}


def _ffprobe_stream_types(video_path):
    """Return the list of codec_type values for every stream, or None if ffprobe is unavailable/failed."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "default=nw=1:nk=1", video_path],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        logger.warning(f"ffprobe failed on {video_path}: {e}")
        return None


def sanitize_video_for_gemini(video_path):
    """If the video has streams Gemini may reject (data/subtitle/attachment/etc.),
    losslessly remux to a temp file keeping only video + audio.

    Returns a tuple (path_to_use, temp_path_to_clean_up_or_None). The caller is
    responsible for deleting temp_path after the upload call returns.
    Gracefully no-ops if ffmpeg/ffprobe aren't available.
    """
    stream_types = _ffprobe_stream_types(video_path)
    if stream_types is None:
        logger.info("ffprobe not available — skipping video sanitize step")
        return video_path, None

    # Keep only video + audio. Anything else (data, subtitle, attachment) is suspect.
    bad_streams = [s for s in stream_types if s not in ("video", "audio")]
    if not bad_streams:
        logger.info(f"Video streams look clean ({stream_types}), no sanitize needed")
        return video_path, None

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning(
            f"Video has non-media streams {bad_streams} that Gemini may reject, "
            f"but ffmpeg is not on PATH — skipping sanitize. "
            f"Install ffmpeg or remux manually if upload fails."
        )
        return video_path, None

    # Remux to a temp file, copy codecs (lossless, fast).
    fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(video_path)[1] or ".mp4",
                                    prefix="gemini_sanitized_")
    os.close(fd)
    logger.info(
        f"Stripping non-media streams {bad_streams} from video via ffmpeg remux → {tmp_path}"
    )
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", video_path,
             "-map", "0:v", "-map", "0:a",
             "-c", "copy", "-movflags", "+faststart",
             tmp_path],
            capture_output=True, text=True, timeout=600, check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"ffmpeg sanitize failed (exit {e.returncode}): {e.stderr[-500:] if e.stderr else ''}. "
            f"Falling back to original file."
        )
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return video_path, None
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg sanitize timed out — falling back to original file")
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return video_path, None

    return tmp_path, tmp_path


def upload_video_for_gemini(client, video_path, poll_timeout=300, poll_interval=2):
    """Upload a video file via the Files API and wait until it's ACTIVE.

    Returns the File object ready to be passed as a content part.
    Raises FileNotFoundError, ValueError, or TimeoutError on failure.
    """
    if not video_path:
        raise ValueError("video path is empty")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in SUPPORTED_VIDEO_EXTS:
        raise ValueError(
            f"Unsupported video extension '{ext}'. Supported: {sorted(SUPPORTED_VIDEO_EXTS)}"
        )

    raw_client = client.client if isinstance(client, UniversalClient) else client
    mime_type = SUPPORTED_VIDEO_EXTS[ext]

    # Strip any non-video/non-audio streams that Gemini's processor may reject.
    upload_path, tmp_to_clean = sanitize_video_for_gemini(video_path)
    file_size = os.path.getsize(upload_path)
    display_name = os.path.basename(upload_path)

    # Log everything that could matter for diagnosing upload failures.
    try:
        import google.genai as _g
        sdk_version = getattr(_g, "__version__", "unknown")
    except Exception:
        sdk_version = "unknown"
    base_url_dbg = get_base_url() or "(default)"
    logger.info(
        f"Uploading video to Gemini Files API: path={upload_path} "
        f"size={file_size} bytes ({file_size / (1024*1024):.2f} MB) "
        f"mime={mime_type} display_name={display_name} "
        f"sdk={sdk_version} base_url={base_url_dbg}"
    )

    # Pass explicit mime_type + display_name via UploadFileConfig.
    # Open as a binary file handle to rule out any path-string ambiguity in older SDKs.
    def _do_upload(file_arg, with_config):
        if with_config:
            cfg = types.UploadFileConfig(mime_type=mime_type, display_name=display_name)
            return raw_client.files.upload(file=file_arg, config=cfg)
        return raw_client.files.upload(file=file_arg)

    uploaded = None
    last_err = None
    for use_handle in (True, False):  # try handle first, then fall back to path
        for with_cfg in (True, False):
            try:
                if use_handle:
                    with open(upload_path, "rb") as fh:
                        uploaded = _do_upload(fh, with_cfg)
                else:
                    uploaded = _do_upload(upload_path, with_cfg)
                logger.info(
                    f"files.upload() returned (handle={use_handle}, config={with_cfg})"
                )
                break
            except (AttributeError, TypeError) as e:
                last_err = e
                logger.warning(
                    f"files.upload() rejected args (handle={use_handle}, config={with_cfg}): {e}"
                )
                continue
        if uploaded is not None:
            break
    if uploaded is None:
        raise RuntimeError(f"All upload call shapes failed; last error: {last_err}")

    # Dump everything Gemini knows about the file we just sent.
    initial_info = {
        "name": getattr(uploaded, "name", None),
        "uri": getattr(uploaded, "uri", None),
        "mime_type": getattr(uploaded, "mime_type", None),
        "size_bytes": getattr(uploaded, "size_bytes", None),
        "display_name": getattr(uploaded, "display_name", None),
        "state": getattr(getattr(uploaded, "state", None), "name", str(getattr(uploaded, "state", ""))),
    }
    logger.info(f"Initial File state from API: {initial_info}")
    if initial_info["size_bytes"] in (None, 0):
        logger.error(
            f"Gemini reports size_bytes={initial_info['size_bytes']} for an upload of "
            f"{file_size} local bytes — the upload body did not reach the API."
        )

    # Gemini has its own copy now — safe to delete the local temp file.
    if tmp_to_clean:
        try:
            os.remove(tmp_to_clean)
            logger.info(f"Cleaned up sanitized temp file: {tmp_to_clean}")
        except OSError as e:
            logger.warning(f"Could not delete temp file {tmp_to_clean}: {e}")

    def _describe_error(file_obj):
        """Pull whatever diagnostic info the File object carries."""
        err = getattr(file_obj, "error", None)
        if err is not None:
            code = getattr(err, "code", None)
            msg = getattr(err, "message", None) or str(err)
            return f"code={code} message={msg}"
        # Some SDKs expose .state_reason / .state_message
        for attr in ("state_reason", "state_message", "status"):
            val = getattr(file_obj, attr, None)
            if val:
                return f"{attr}={val}"
        return "no error details returned by API"

    deadline = time.time() + poll_timeout
    while True:
        state = getattr(getattr(uploaded, "state", None), "name", None) or str(getattr(uploaded, "state", ""))
        if state == "ACTIVE":
            logger.info(f"Video file is ACTIVE: {uploaded.name}")
            return uploaded
        if state == "FAILED":
            detail = _describe_error(uploaded)
            # Dump every attribute on the File object so we can see exactly what
            # Gemini received and complained about.
            try:
                dump = {k: getattr(uploaded, k) for k in dir(uploaded)
                        if not k.startswith("_") and not callable(getattr(uploaded, k, None))}
                # Convert non-stringable values
                dump_safe = {k: (str(v)[:300] if v is not None else None) for k, v in dump.items()}
                logger.error(f"Full File object on FAILED state: {dump_safe}")
            except Exception as dump_err:
                logger.error(f"Could not dump File object: {dump_err}")
            raise RuntimeError(
                f"Video upload FAILED for {video_path}: {detail}. "
                f"(local size={file_size / (1024*1024):.1f} MB, "
                f"api size_bytes={getattr(uploaded, 'size_bytes', '?')}, mime={mime_type}). "
                f"See log above for full File object dump."
            )
        if time.time() > deadline:
            raise TimeoutError(
                f"Video upload did not become ACTIVE within {poll_timeout}s (last state: {state})"
            )
        logger.info(f"Waiting for video to process (state: {state})...")
        time.sleep(poll_interval)
        uploaded = raw_client.files.get(name=uploaded.name)


def _build_thinking_config(thinking_level):
    """Build a ThinkingConfig that works on both old and new google-genai SDKs.

    Newer SDKs accept `thinking_level` (str: minimal/low/medium/high) for Gemini 3.
    Older SDKs only accept `thinking_budget` (int) — map the level to a budget.

    A thinking_level of "none" means "send no thinking config" — the model uses
    its own default. (The API has no explicit off switch below "minimal".)
    """
    if types is None or thinking_level == "none":
        return None
    supported = set(getattr(types.ThinkingConfig, "model_fields", {}).keys())
    if "thinking_level" in supported:
        return types.ThinkingConfig(thinking_level=thinking_level)
    # Fallback: map level → integer budget for older SDKs.
    # -1 = dynamic (max), 0 = disabled where allowed.
    budget_map = {"minimal": 0, "low": 1024, "medium": 8192, "high": -1}
    return types.ThinkingConfig(thinking_budget=budget_map.get(thinking_level, 0))


from .env_utils import get_api_key, get_base_url, get_effective_api_key
from .utils import ChatHistory
from .image_utils import (
    create_placeholder_image,
    prepare_batch_images,
    process_images_for_comfy,
)
from .response_utils import prepare_response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logging from the Google API client
# This prevents API keys from being exposed in HTTP request logs and HTML spam
logging.getLogger("httpx").setLevel(logging.ERROR)  # More restrictive
logging.getLogger("httpcore").setLevel(logging.ERROR)  # More restrictive
logging.getLogger("google.genai").setLevel(logging.ERROR)  # More restrictive
logging.getLogger("google.auth").setLevel(logging.ERROR)  # More restrictive
logging.getLogger("urllib3").setLevel(logging.ERROR)  # Add urllib3
logging.getLogger("requests").setLevel(logging.ERROR)  # Add requests

gemini_models = [
            "gemini-3-pro-image-preview",
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-image-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash-preview",
            ]
gemini_image_models = [
                "gemini-3-pro-image-preview",
                "gemini-3.1-flash-image-preview",
            ]


def generate_consistent_seed(input_seed=0, use_random=False):
    """
    Generate a consistent seed for the Gemini API.
    
    This function uses a more reliable approach to seed generation:
    - If input_seed is non-zero and use_random is False, use the input_seed
    - Otherwise, generate a high-quality random seed based on uuid and time
    
    Returns:
        int: A seed value within the INT32 range (0 to 2^31-1)
    """
    max_int32 = 2**31 - 1
    
    if input_seed != 0 and not use_random:
        # Use the provided seed, but ensure it's within INT32 range
        adjusted_seed = input_seed % max_int32
        logger.info(f"Using provided seed (adjusted to INT32 range): {adjusted_seed}")
        return adjusted_seed
    
    # For random seeds, use a more robust method that won't collide with ComfyUI's seed generation
    # Create a unique identifier by combining:
    # 1. A UUID (universally unique)
    # 2. Current high-precision time
    # 3. ComfyUI's random seed (if we wanted to use it)
    
    unique_id = str(uuid.uuid4())
    current_time = str(time.time_ns())  # Nanosecond precision
    random_component = str(random.randint(0, max_int32))
    
    # Combine and hash all components to get a deterministic but high-quality random value
    combined = unique_id + current_time + random_component
    hash_hex = hashlib.md5(combined.encode()).hexdigest()
    
    # Convert first 8 characters of hash to integer and ensure within INT32 range
    hash_int = int(hash_hex[:8], 16) % max_int32
    
    logger.info(f"Generated random seed: {hash_int}")
    return hash_int


def create_appropriate_client(api_key, api_key_source="unknown"):
    """
    Create Gemini client
    
    Args:
        api_key: The API key to use for authentication
        api_key_source: Source type of the API key
        
    Returns:
        Configured Gemini client
    """
    return create_gemini_client(api_key, api_key_source), "gemini"


def create_gemini_client(api_key, api_key_source="unknown"):
    """
    Create Gemini client with configurable base URL
    
    Args:
        api_key: The API key to use for authentication
        api_key_source: Source type of the API key
        
    Returns:
        Configured Gemini client
    """
    
    # Mask the API key for logging
    masked_key = api_key[:5] + "..." if len(api_key) > 5 else "****"
    
    base_url = get_base_url()
    
    if base_url:
        # Use custom base URL if provided
        logger.info(f"Creating Gemini client with custom base URL: {base_url} (API key: {masked_key}, source: {api_key_source})")
        
        # Create HTTP options with custom base URL
        http_options = types.HttpOptions(base_url=base_url)
        
        client = genai.Client(
            api_key=api_key,
            http_options=http_options
        )
    else:
        # Use default endpoint
        logger.info(f"Creating Gemini client with default endpoint (API key: {masked_key}, source: {api_key_source})")
        client = genai.Client(api_key=api_key)

    return client


def create_vertex_client(api_key=None, project=None, location=None, api_key_source="vertex"):
    """
    Create a Vertex AI-backed genai client.

    Uses the same google-genai SDK as the AI Studio path, just pointed at
    Vertex (vertexai=True). The downstream generate_content / generate_images
    calls are identical, so this client is interchangeable with the AI Studio
    one everywhere UniversalClient is used.

    Args:
        api_key: Vertex "express mode" API key (optional if using project/location).
        project: GCP project id (optional in express mode).
        location: GCP region, e.g. "us-central1" or "global" (optional in express mode).
        api_key_source: Source label, for logging only.

    Returns:
        Configured genai.Client with vertexai=True.
    """
    kwargs = {"vertexai": True}

    # google-genai treats api_key and project/location as MUTUALLY EXCLUSIVE.
    # Express mode: pass only the api_key (it already encodes project+region).
    # ADC mode: pass project+location and let google.auth resolve credentials.
    if api_key:
        kwargs["api_key"] = api_key
    else:
        if project:
            kwargs["project"] = project
        if location:
            kwargs["location"] = location

    masked_key = (api_key[:5] + "...") if api_key and len(api_key) > 5 else ("****" if api_key else "(none)")
    if api_key:
        logger.info(
            f"Creating Vertex AI client in express mode (key: {masked_key}, source: {api_key_source}); "
            f"project/location are ignored in this mode."
        )
    else:
        logger.info(
            f"Creating Vertex AI client via ADC (project: {project or '(none)'}, "
            f"location: {location or '(none)'}, source: {api_key_source})"
        )

    return genai.Client(**kwargs)


class UniversalClient:
    """
    Universal client wrapper for Gemini client
    """
    
    def __init__(self, client, client_type: str):
        self.client = client
        self.client_type = client_type
        self.logger = logging.getLogger(__name__)
    
    def generate_content(self, contents, model_name: str, generation_config=None, **kwargs):
        """Generate content using Gemini client"""
        return self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=generation_config,
            **kwargs
        )
    
    def chats(self):
        """Return chat interface"""
        return self.client.chats
    
    def models(self):
        """Return models interface"""
        return self.client.models


class Soze_GeminiAdvanced:
    def __init__(self):
        self.api_key = ""
        self.chat_history = ChatHistory()
        self.last_external_api_key = ""  # Track the last external API key
        self.api_key_source = None  # Track where the API key came from
        
        # First check system environment variables
        system_api_key = os.environ.get("GEMINI_API_KEY", "")
        if system_api_key:
            self.api_key = system_api_key
            self.api_key_source = "system environment variables"
            logger.info("Successfully loaded Gemini API key from system environment")
        else:
            # Next, try to directly check shell configuration files
            home_dir = os.path.expanduser("~")
            shell_config_files = [
                os.path.join(home_dir, ".zshrc"),
                os.path.join(home_dir, ".bashrc"),
                os.path.join(home_dir, ".bash_profile")
            ]
            
            import re
            shell_key = None
            shell_source = None
            
            for config_file in shell_config_files:
                if os.path.exists(config_file):
                    logger.debug(f"Checking {config_file} for API key...")
                    try:
                        with open(config_file, 'r') as f:
                            content = f.read()
                            # Look for export VAR=value or VAR=value patterns
                            patterns = [
                                r'export\s+GEMINI_API_KEY=[\'\"]?([^\s\'\"]+)[\'\"]?',
                                r'GEMINI_API_KEY=[\'\"]?([^\s\'\"]+)[\'\"]?'
                            ]
                            for pattern in patterns:
                                matches = re.findall(pattern, content)
                                if matches:
                                    shell_key = matches[0]
                                    shell_source = os.path.basename(config_file)
                                    logger.info(f"Found Gemini API key in {shell_source}")
                                    # Also set in environment for future use
                                    os.environ["GEMINI_API_KEY"] = shell_key
                                    break
                    except Exception as e:
                        logger.error(f"Error reading {config_file}: {str(e)}")
                if shell_key:
                    break
                    
            if shell_key:
                self.api_key = shell_key
                self.api_key_source = shell_source
                logger.info(f"Successfully loaded Gemini API key from {shell_source}")
            else:
                # Last resort: check .env files
                env_api_key = get_api_key("GEMINI_API_KEY", "Gemini")
                if env_api_key:
                    self.api_key = env_api_key
                    self.api_key_source = ".env file"
                    logger.info("Successfully loaded Gemini API key from .env file")
                else:
                    logger.warning("No Gemini API key found in any location (system env, shell configs, .env). You'll need to provide it in the node.")
        
        # Log key information (masked for security)
        if self.api_key:
            masked_key = self.api_key[:5] + "..." if len(self.api_key) > 5 else "****"
            logger.info(f"Using Gemini API key ({masked_key}) from {self.api_key_source}")
        
        # Check for Google Generative AI SDK
        self.genai_available = self._check_genai_availability()

    def _check_genai_availability(self):
        """Check if Google Generative AI SDK is available"""
        try:
            # Import just to check availability
            from google import genai

            return True
        except ImportError:
            logger.error(
                "Google Generative AI SDK not installed. Install with: pip install google-genai"
            )
            return False

    # --- Backend hooks (overridden by the Vertex subclass) ----------------
    # These three methods are the only backend-specific seams. The AI Studio
    # implementation lives here; Soze_VertexGemini overrides them to talk to
    # Vertex AI while reusing all of the generation logic below.

    def _resolve_api_key(self, external_api_key=""):
        """Return (api_key, source) for the AI Studio backend, or (None, None)."""
        cleaned = external_api_key.strip() if external_api_key else ""
        if cleaned:
            logger.info("Using external Gemini API key")
            return cleaned, "external"

        api_key, _ = get_effective_api_key()
        if api_key:
            return api_key, "environment"

        if self.last_external_api_key:
            logger.info("Using previously provided external API key")
            return self.last_external_api_key, "cached"

        return None, None

    def _no_key_error(self):
        """Human-readable error when no usable credential was found."""
        return (
            "ERROR: No Gemini API key found. Please set GEMINI_API_KEY in your environment "
            "or provide it in the external_api_key field."
        )

    def _make_client(self, api_key, api_key_source="unknown"):
        """Return (raw_client, client_type) for the AI Studio backend."""
        return create_appropriate_client(api_key, api_key_source)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Create a vivid word-picture representation of this image include elements that characterize the subject, costume, prop elemts, the action, the background, layout and composition elements present on the scene, be sure to mention the style and mood of the scene. Like it would a film director or director of photography"}),
                "operation_mode": (
                    ["analysis", "generate_text", "generate_images"],
                    {"default": "generate_images"},
                ),
                "model_name": (
                    gemini_models,
                    {"default": gemini_models[0]},
                ),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "thinking_level": (
                    ["minimal", "low", "medium", "high"],
                    {"default": "minimal"},
                ),
                "media_resolution": (
                    ["1K", "2K", "4K"],
                    {"default": "1K"},
                ),
            },
            "optional": {
                "images": ("IMAGE",),
                "video": ("STRING", {"default": "", "multiline": False}),
                "frames": ("IMAGE",),
                "audio": ("AUDIO",),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "sequential_generation": ("BOOLEAN", {"default": False}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 20}),
                "aspect_ratio": (
                    ["none", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                    {"default": "none"},
                ),
                "external_api_key": ("STRING", {"default": ""}),
                "bypass": ("BOOLEAN", {"default": False}),
                "chat_mode": ("BOOLEAN", {"default": False}),
                "clear_history": ("BOOLEAN", {"default": False}),
                "structured_output": ("BOOLEAN", {"default": False}),
                "max_images": ("INT", {"default": 6, "min": 1, "max": 16}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768}),
                "use_random_seed": ("BOOLEAN", {"default": False}),
                "api_call_delay": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "image")
    FUNCTION = "generate_content"
    CATEGORY = "Soze Nodes"

    # Image-capable model IDs for this backend. Overridden by the Vertex
    # subclass, since Vertex uses different model id strings than AI Studio.
    IMAGE_MODELS = gemini_image_models

    def generate_content(
        self,
        prompt,
        operation_mode="analysis",
        model_name=gemini_models[0],
        temperature=0.4,
        thinking_level="minimal",
        media_resolution="medium",
        images=None,
        video="",
        frames=None,
        audio=None,
        seed=0,
        sequential_generation=False,
        batch_count=1,
        aspect_ratio="none",
        external_api_key="",
        bypass=False,
        chat_mode=False,
        clear_history=False,
        structured_output=False,
        max_images=6,
        max_output_tokens=8192,
        use_random_seed=False,
        api_call_delay=1.0,
        system_prompt="",
    ):
        """Generate content using Gemini model with various input types."""

        # Bypass: short-circuit without any API call or processing.
        if bypass:
            logger.info("Soze_GeminiAdvanced: bypass=True — skipping all processing")
            return ("", None)

        # Check if Google Generative AI SDK is available
        if not self.genai_available:
            return (
                "ERROR: Google Generative AI SDK not installed. Install with: pip install google-genai",
                create_placeholder_image(),
            )

        # Handle chat mode clearing if requested
        if clear_history and chat_mode:
            self.chat_history.clear()
            logger.info("Chat history cleared")

        # API key resolution (backend-specific via _resolve_api_key)
        cleaned_external_key = external_api_key.strip() if external_api_key else ""
        api_key, api_key_source = self._resolve_api_key(external_api_key)

        if not api_key:
            return (self._no_key_error(), create_placeholder_image())
        # Generate a consistent seed for this operation
        operation_seed = generate_consistent_seed(seed, use_random_seed)

        # Handle image generation mode
        if operation_mode == "generate_images":
            return self.generate_images(
                prompt=prompt,
                model_name=model_name,
                images=images,
                batch_count=batch_count,
                temperature=temperature,
                thinking_level=thinking_level,
                media_resolution=media_resolution,
                seed=operation_seed,
                max_images=max_images,
                aspect_ratio=aspect_ratio,
                use_random_seed=use_random_seed,
                external_api_key=cleaned_external_key,
                api_key_source=api_key_source,
                sequential_generation=sequential_generation,
                api_call_delay=api_call_delay,
            )

        # Initialize the API client with the API key
        try:
            raw_client, client_type = self._make_client(api_key, api_key_source)
            client = UniversalClient(raw_client, client_type)
            logger.info(f"Using Gemini client for API calls")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error initializing Gemini client: {error_msg}", exc_info=True)
            
            # Truncate very long error messages to prevent log spam
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "... [truncated]"
            
            # Provide helpful messages for common errors
            if "invalid api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                return (
                    "ERROR: Invalid Gemini API key. Please check your API key and try again.",
                    create_placeholder_image(),
                )
            elif "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                return (
                    "ERROR: API quota exceeded. You've reached your usage limit for the Gemini API.",
                    create_placeholder_image(),
                )
                
            return (f"Error initializing Gemini client: {error_msg}", create_placeholder_image())

        # Configure safety settings and generation parameters
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Prepare generation config with all necessary parameters
        generation_config_params = {
            "max_output_tokens": max_output_tokens, 
            "temperature": temperature, 
            "seed": operation_seed,
            "safety_settings": safety_settings
        }

        # Add system instruction if provided
        if system_prompt:
            generation_config_params["system_instruction"] = system_prompt
        
        # Add Thinking Level for Gemini 3 models
        generation_config_params["thinking_config"] = _build_thinking_config(thinking_level)
        logger.info(f"Using thinking_level: {thinking_level}")
        
        # Add Media Resolution for Gemini 3 models (affects image/video processing)
        if media_resolution != "medium":  # medium is default
            # Note: Media resolution is typically handled at the media level, not generation config
            # This might need to be applied when processing media inputs
            logger.info(f"Media resolution set to: {media_resolution}")
        
        # Add response_mime_type for structured output
        if structured_output:
            generation_config_params["response_mime_type"] = "application/json"
            logger.info("Requesting structured JSON output")
            
        generation_config = types.GenerateContentConfig(**generation_config_params)

        # If a video file path was supplied, upload it once and reuse the File object.
        uploaded_video_file = None
        video_path = video.strip() if isinstance(video, str) else ""
        if video_path:
            try:
                uploaded_video_file = upload_video_for_gemini(client, video_path)
            except Exception as upload_err:
                logger.error(f"Video upload failed: {upload_err}", exc_info=True)
                return (f"Error uploading video: {upload_err}", create_placeholder_image())

        try:
            if chat_mode:
                # Handle chat mode with proper history
                history = self.chat_history.get_messages_for_api()

                if uploaded_video_file is not None:
                    # Build parts directly: prompt text + uploaded video file.
                    current_message_parts = [
                        types.Part(text=prompt),
                        types.Part(file_data=types.FileData(
                            file_uri=uploaded_video_file.uri,
                            mime_type=uploaded_video_file.mime_type,
                        )),
                    ]
                    contents = None  # not used in this branch
                else:
                    # Create appropriate content parts based on input type
                    contents = prepare_response(
                        prompt,
                        "image" if images is not None else ("video" if frames is not None else ("audio" if audio is not None else "text")),
                        None,
                        images,
                        frames,
                        audio,
                        max_images,
                    )
                
                # Extract content for chat format (skip if already set by video-file branch)
                if uploaded_video_file is None:
                    if (
                        isinstance(contents, list)
                        and len(contents) == 1
                        and isinstance(contents[0], dict)
                        and "parts" in contents[0]
                    ):
                        current_message_parts = contents[0]["parts"]
                    else:
                        # Fallback if we get unexpected format
                        current_message_parts = [{"text": prompt}]
                        logger.warning("Unexpected format from prepare_response, using text prompt only")

                try:
                    # Create chat session with proper history (Gemini)
                    chat_session = client.chats().create(
                        model=model_name,
                        history=history
                    )

                    # Send message to chat and get response
                    response = chat_session.send_message(
                        content=current_message_parts,
                        config=generation_config,
                    )

                    # Process the response
                    if structured_output:
                        try:
                            # Try to parse the response as JSON
                            import json
                            raw_text = response.text
                            parsed_json = json.loads(raw_text)
                            # Pretty print the JSON for better readability
                            response_text = json.dumps(parsed_json, indent=2)
                            logger.info("Successfully parsed structured JSON chat response")
                        except (json.JSONDecodeError, Exception) as e:
                            # Fallback to raw text if JSON parsing fails
                            logger.warning(f"Failed to parse structured JSON output in chat: {str(e)}")
                            response_text = f"Warning: Requested JSON output but received non-JSON response:\n\n{response.text}"
                    else:
                        response_text = response.text

                    # Add to history and format response
                    self.chat_history.add_message("user", prompt)
                    self.chat_history.add_message("assistant", response_text)

                    # Return the chat history
                    generated_content = self.chat_history.get_formatted_history()
                    
                except Exception as chat_error:
                    logger.error(f"Error in chat session: {str(chat_error)}", exc_info=True)
                    if "not supported" in str(chat_error).lower() and "json" in str(chat_error).lower():
                        generated_content = f"Error: This model doesn't support structured JSON output in chat mode. Try a different model or disable structured output."
                    else:
                        generated_content = f"Error in chat session: {str(chat_error)}"
                    # Add error to chat history for transparency
                    self.chat_history.add_message("user", prompt)
                    self.chat_history.add_message("assistant", generated_content)

            else:
                # Standard non-chat mode - prepare content for each input type
                if uploaded_video_file is not None:
                    contents = [types.Content(role="user", parts=[
                        types.Part(text=prompt),
                        types.Part(file_data=types.FileData(
                            file_uri=uploaded_video_file.uri,
                            mime_type=uploaded_video_file.mime_type,
                        )),
                    ])]
                else:
                    contents = prepare_response(
                        prompt,
                        "image" if images is not None else ("video" if frames is not None else "text"),
                        None,
                        images,
                        frames,
                        audio,
                        max_images,
                    )

                # Generate content using the model
                response = client.generate_content(
                    contents=contents,
                    model_name=model_name,
                    generation_config=generation_config,
                )

                # Process the response, handling structured output if requested
                if structured_output:
                    try:
                        # Try to parse the response as JSON
                        import json
                        raw_text = response.text
                        parsed_json = json.loads(raw_text)
                        # Pretty print the JSON for better readability
                        generated_content = json.dumps(parsed_json, indent=2)
                        logger.info("Successfully parsed structured JSON response")
                    except (json.JSONDecodeError, Exception) as e:
                        # Fallback to raw text if JSON parsing fails
                        logger.warning(f"Failed to parse structured JSON output: {str(e)}")
                        generated_content = f"Warning: Requested JSON output but received non-JSON response:\n\n{response.text}"
                else:
                    # Standard text response
                    generated_content = response.text

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error generating content: {error_msg}")
            
            # Truncate very long error messages to prevent log spam
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "... [truncated]"
            generated_content = f"Error: {error_msg}"
            raise e  # Re-raise the exception after logging

        # For analysis mode, return the text response and an empty placeholder image
        return (generated_content, create_placeholder_image())

    def generate_images(
        self,
        prompt,
        model_name,
        images=None,
        batch_count=1,
        temperature=0.4,
        thinking_level="minimal",
        media_resolution="medium",
        seed=0,
        max_images=6,
        aspect_ratio="none",
        use_random_seed=False,
        external_api_key="",
        api_key_source=None,
        sequential_generation=False,
        api_call_delay=1.0,
    ):
        """Generate images using Gemini models with image generation capabilities"""
        try:
            # Import here to avoid ImportError during ComfyUI startup
            # This determination of `client_type` must happen before the model name is validated.
            # We peek ahead to see what client will be created.
            # For Gemini, model names should be image-capable
            image_capable_models = self.IMAGE_MODELS
            if model_name not in image_capable_models:
                original_model = model_name
                model_name = image_capable_models[0]  # A safe default for Gemini
                logger.warning(
                    f"Model '{original_model}' may not support image generation. "
                    f"Switched to '{model_name}'."
                )

            # Resolve the credential via the backend hook (AI Studio or Vertex)
            api_key, resolved_source = self._resolve_api_key(external_api_key)
            if api_key_source is None:
                api_key_source = resolved_source

            if not api_key:
                return (self._no_key_error(), create_placeholder_image())

            # Create the client (AI Studio or Vertex, per the backend hook)
            try:
                raw_client, client_type = self._make_client(api_key, api_key_source)
                client = UniversalClient(raw_client, client_type)
                logger.info("Using Gemini client for image generation")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error initializing Gemini client for image generation: {error_msg}", exc_info=True)
                
                # Truncate very long error messages to prevent log spam
                if len(error_msg) > 500:
                    error_msg = error_msg[:500] + "... [truncated]"
                
                # Provide more helpful messages for common errors
                if "invalid api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                    return (
                        "ERROR: Invalid Gemini API key. Please check your API key and try again.",
                        create_placeholder_image(),
                    )
                elif "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
                    return (
                        "ERROR: API quota exceeded. You've reached your usage limit for the Gemini API.",
                        create_placeholder_image(),
                    )
                
                return (f"Error initializing Gemini client: {error_msg}", create_placeholder_image())

            # Use the same seed as passed from generate_content to ensure consistency
            logger.info(f"Using seed for image generation: {seed}")

            # Define aspect ratio dimensions for Imagen 3
            aspect_ratio_dimensions = {
                "none": (1024, 1024),  # Default square format
                "1:1": (1024, 1024),  # Square
                "16:9": (1408, 768),  # Landscape widescreen
                "9:16": (768, 1408),  # Portrait widescreen
                "4:3": (1280, 896),  # Standard landscape
                "3:4": (896, 1280),  # Standard portrait
                "5:4": (1024, 819),  # Medium landscape format
                "4:5": (819, 1024),  # Medium portrait format
                "2:3": (836, 1254),
                "3:2": (1254, 836),
                "21:9": (1536, 640),
            }

            # Get target dimensions based on aspect ratio
            target_width, target_height = aspect_ratio_dimensions.get(aspect_ratio, (1024, 1024))
            logger.info(f"Using resolution {target_width}x{target_height} for aspect ratio {aspect_ratio}")

            all_generated_images_bytes = []
            all_generated_text = []
            status_text = ""

            # Set up generation config with required fields
            gen_config_args = {
                "temperature": temperature,
                "response_modalities": [types.Modality.TEXT, types.Modality.IMAGE] if hasattr(types, "Modality") else ["TEXT", "IMAGE"],
                "seed": seed,  # Always include seed in config
                "safety_settings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            }
            
            # Add Image Config if appropriate
            using_image_config = False
            image_config_args = {}
            
            if aspect_ratio != "none":
                image_config_args["aspect_ratio"] = aspect_ratio
                
            # Handle resolution/image_size
            if media_resolution in ["1K", "2K", "4K"]:
                image_config_args["image_size"] = media_resolution
                
            if image_config_args and hasattr(types, "ImageConfig"):
                try:
                    # Note: Using aspect_ratio from the input directly which matches API expectations (e.g. "16:9")
                    gen_config_args["image_config"] = types.ImageConfig(**image_config_args)
                    using_image_config = True
                    logger.info(f"Using ImageConfig: {image_config_args}")
                except Exception as e:
                    logger.warning(f"Failed to create ImageConfig: {e}. Falling back to prompt engineering.")
            
            # Add Thinking Level for Gemini 3 models
            gen_config_args["thinking_config"] = _build_thinking_config(thinking_level)
            logger.info(f"Using thinking_level: {thinking_level} for image generation")

            generation_config = types.GenerateContentConfig(**gen_config_args)

            # Process reference images if provided
            ref_images = []
            if images is not None and isinstance(images, torch.Tensor) and images.nelement() > 0:
                # Determine max size based on media resolution setting
                media_resolution_sizes = {
                    "low": 512, "1K": 1024,
                    "medium": 1024, "2K": 2048,
                    "high": 2048, "4K": 4096,
                    "ultra_high": 4096
                }
                max_input_size = media_resolution_sizes.get(media_resolution, 1024)
                # Use the larger of target dimensions or media resolution limit
                effective_max_size = max(max_input_size, max(target_width, target_height))
                
                # Convert tensor to list of PIL images - resize to match target dimensions
                ref_images = prepare_batch_images(images, max_images, max_size=effective_max_size)
                logger.info(f"Prepared {len(ref_images)} reference images with media resolution: {media_resolution} (max size: {effective_max_size})")

            # Initialize collections for results - accumulate all results before returning
                
            # Sequential generation mode
            if sequential_generation:
                logger.info(f"Using sequential generation mode for {batch_count} steps")
                    
                # Initialize history for sequential generation
                history = []
                current_prompt = prompt
                    
                # Process each step in the sequence
                for i in range(batch_count):
                    # Add delay before each API call except the first one
                    if i > 0 and api_call_delay > 0:
                        logger.info(f"Waiting for {api_call_delay:.1f} seconds before next API call...")
                        time.sleep(api_call_delay)
                        
                    try:
                        # Generate a unique seed for each step
                        current_seed = (seed + i) % (2**31 - 1)
                        logger.info(f"Sequential step {i+1}/{batch_count} with seed {current_seed}")
                            
                        # Update config with current seed
                        # Use gen_config_args as base to preserve thinking settings and other parameters
                        step_args = gen_config_args.copy()
                        step_args["seed"] = current_seed
                        step_config = types.GenerateContentConfig(**step_args)
                            
                        # Prepare content for this step
                        if i == 0:
                            parts = []
                            # First step includes reference images if available
                            if ref_images:
                                # Construct prompt with specific dimensions
                                # Only add dimensions to prompt if NOT using ImageConfig
                                aspect_string = ""
                                if aspect_ratio != "none" and not using_image_config:
                                    aspect_string = f" with dimensions {target_width}x{target_height}"
                                    
                                content_text = f"Generate a sequence of {batch_count} images{aspect_string}. First image: {prompt}"
                                parts.append(types.Part(text=content_text))
                                    
                                for img in ref_images:
                                    img_byte_arr = BytesIO()
                                    img.save(img_byte_arr, format='PNG')
                                    img_byte_arr = img_byte_arr.getvalue()
                                    parts.append(
                                        types.Part(
                                            inline_data=types.Blob(
                                                mime_type="image/png",
                                                data=img_byte_arr
                                            )
                                        )
                                    )
                            else:
                                # Include specific dimensions in the prompt
                                # Only add dimensions to prompt if NOT using ImageConfig
                                if aspect_ratio != "none" and not using_image_config:
                                    content_text = f"Generate a sequence of {batch_count} images with dimensions {target_width}x{target_height}. First image: {prompt}"
                                else:
                                    content_text = f"Generate a sequence of {batch_count} images. First image: {prompt}"
                                parts.append(types.Part(text=content_text))
                                    
                            # Use types.Content for content
                            content = [types.Content(role="user", parts=parts)]
                                
                            # Add proper Content object to history
                            history.append(content[0])

                        else:
                            # Subsequent steps - continue from previous output
                            content_text = f"Generate the next image in the sequence. Step {i+1} of {batch_count}: {current_prompt}"
                            parts = [types.Part(text=content_text)]
                            content = [types.Content(role="user", parts=parts)]
                            history.append(content[0])
                            
                        # Generate content for this step
                        response = client.generate_content(
                            contents=content if i == 0 else history, 
                            model_name=model_name, 
                            generation_config=step_config
                        )
                            
                        # Process response
                        step_images_bytes = []
                        step_text = ""
                        finish_reason = None
                        safety_info = ""
                        prompt_feedback_info = ""
                        
                        # Check for prompt feedback (may contain block reason)
                        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                            pf = response.prompt_feedback
                            if hasattr(pf, 'block_reason') and pf.block_reason:
                                prompt_feedback_info += f"Block reason: {pf.block_reason}. "
                            if hasattr(pf, 'safety_ratings') and pf.safety_ratings:
                                ratings = [f"{r.category.name}:{r.probability.name}" for r in pf.safety_ratings if hasattr(r, 'category') and hasattr(r, 'probability')]
                                if ratings:
                                    prompt_feedback_info += f"Prompt safety: {', '.join(ratings)}. "
                            
                        if hasattr(response, 'candidates') and response.candidates:
                            for candidate in response.candidates:
                                if hasattr(candidate, 'finish_reason'):
                                    finish_reason = candidate.finish_reason
                                    logger.info(f"Step {i+1} finish reason: {finish_reason}")
                                
                                # Extract safety ratings from candidate
                                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                                    ratings = [f"{r.category.name}:{r.probability.name}" for r in candidate.safety_ratings if hasattr(r, 'category') and hasattr(r, 'probability')]
                                    if ratings:
                                        safety_info = f"Safety ratings: {', '.join(ratings)}. "
                                    
                                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts is not None:
                                    model_parts = []
                                    for part in candidate.content.parts:
                                        # Extract text
                                        if hasattr(part, 'text') and part.text:
                                            # Skip accidental HTML payloads
                                            if ("<!DOCTYPE html" in part.text) or ("<html" in part.text.lower()):
                                                pass
                                            else:
                                                step_text += part.text + "\n"
                                            model_parts.append({"text": part.text})
                                            
                                        # Extract image data
                                        if hasattr(part, 'inline_data') and part.inline_data:
                                            try:
                                                image_binary = part.inline_data.data
                                                step_images_bytes.append(image_binary)
                                                model_parts.append({
                                                    "inline_data": {
                                                        "mime_type": part.inline_data.mime_type,
                                                        "data": image_binary
                                                    }
                                                })
                                            except Exception as img_error:
                                                logger.error(f"Error extracting image from response: {str(img_error)}")
                                        
                                    # Add model response to history
                                    model_response_parts = []
                                    for part_dict in model_parts:
                                        if "text" in part_dict:
                                            model_response_parts.append(types.Part(text=part_dict["text"]))
                                        elif "inline_data" in part_dict:
                                            model_response_parts.append(types.Part(
                                                inline_data=types.Blob(
                                                    mime_type=part_dict["inline_data"]["mime_type"],
                                                    data=part_dict["inline_data"]["data"]
                                                )
                                            ))
                                    history.append(types.Content(role="model", parts=model_response_parts))
                            
                        # Update current prompt for next iteration with text from this response
                        if step_text.strip():
                            # Use the text response as context for the next prompt
                            current_prompt = step_text.strip()
                            all_generated_text.append(f"Step {i+1}:\n{current_prompt}")
                        else:
                            all_generated_text.append(f"Step {i+1}: No text generated")
                            # Use a generic continue prompt if no text was generated
                            current_prompt = "Continue the sequence"
                            
                        # Accumulate images from this step (don't process them yet)
                        if step_images_bytes:
                            all_generated_images_bytes.extend(step_images_bytes)
                            status_text += f"Step {i+1} (seed {current_seed}): Generated {len(step_images_bytes)} image(s)\n"
                        else:
                            finish_info = f" (Reason: {finish_reason})" if finish_reason else ""
                            status_text += f"Step {i+1} (seed {current_seed}): No images generated{finish_info}\n"
                            
                            # Include additional diagnostic info
                            if prompt_feedback_info:
                                status_text += f"Prompt feedback: {prompt_feedback_info}\n"
                            if safety_info:
                                status_text += f"{safety_info}\n"
                                
                            # If no images were generated in this step, we might want to stop the sequence
                            if finish_reason and "SAFETY" in str(finish_reason).upper():
                                status_text += "Step blocked for safety reasons. Stopping sequence.\n"
                                break
                        
                    except Exception as batch_error:
                        error_msg = f"Error in sequence step {i+1}: {str(batch_error)}"
                        logger.error(error_msg, exc_info=True)
                        status_text += error_msg + "\n"
                        # Continue with next step despite error
                
            # Standard batch generation mode
            else:
                # Handle standard batch generation of separate images
                for i in range(batch_count):
                    # Add delay before each API call except the first one
                    if i > 0 and api_call_delay > 0:
                        logger.info(f"Waiting for {api_call_delay:.1f} seconds before next API call...")
                        time.sleep(api_call_delay)
                        
                    try:
                        # Generate a unique seed for each batch based on the operation seed
                        # This ensures consistent but different seeds across batches
                        current_seed = (seed + i) % (2**31 - 1)
                            
                        # Create batch-specific configuration with the unique seed
                        # Use gen_config_args as base to preserve thinking settings and other parameters
                        batch_args = gen_config_args.copy()
                        batch_args["seed"] = current_seed
                        batch_config = types.GenerateContentConfig(**batch_args)

                        # Log the seed being used
                        logger.info(f"Generating batch {i+1} with seed {current_seed}")
                            
                        # Prepare content for batch
                        parts = []
                        if ref_images:
                            # Construct prompt with specific dimensions
                            # Only add dimensions to prompt if NOT using ImageConfig
                            aspect_string = ""
                            if aspect_ratio != "none" and not using_image_config:
                                aspect_string = f" with dimensions {target_width}x{target_height}"
                                    
                            content_text = f"Generate a new image{aspect_string}: {prompt}"
                            parts.append(types.Part(text=content_text))
                                
                            for img in ref_images:
                                img_byte_arr = BytesIO()
                                img.save(img_byte_arr, format='PNG')
                                img_byte_arr = img_byte_arr.getvalue()
                                parts.append(
                                    types.Part(
                                        inline_data=types.Blob(
                                            mime_type="image/png",
                                            data=img_byte_arr
                                        )
                                    )
                                )
                        else:
                            # Include specific dimensions in the prompt
                            # Only add dimensions to prompt if NOT using ImageConfig
                            if aspect_ratio != "none" and not using_image_config:
                                content_text = f"Generate a detailed, high-quality image with dimensions {target_width}x{target_height} of: {prompt}"
                            else:
                                content_text = f"Generate a detailed, high-quality image of: {prompt}"
                            parts.append(types.Part(text=content_text))

                        # Construct proper Content object
                        content = [types.Content(role="user", parts=parts)]

                        # Generate content
                        response = client.generate_content(
                            contents=content, 
                            model_name=model_name, 
                            generation_config=batch_config
                        )

                        # Process response to extract generated images and text
                        batch_images_bytes = []
                        response_text = ""
                        finish_reason = None
                        safety_info = ""
                        prompt_feedback_info = ""

                        # Check for prompt feedback (may contain block reason)
                        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                            pf = response.prompt_feedback
                            if hasattr(pf, 'block_reason') and pf.block_reason:
                                prompt_feedback_info += f"Block reason: {pf.block_reason}. "
                            if hasattr(pf, 'safety_ratings') and pf.safety_ratings:
                                ratings = [f"{r.category.name}:{r.probability.name}" for r in pf.safety_ratings if hasattr(r, 'category') and hasattr(r, 'probability')]
                                if ratings:
                                    prompt_feedback_info += f"Prompt safety: {', '.join(ratings)}. "

                        # Check for finish reason which might explain why no images were generated
                        if hasattr(response, 'candidates') and response.candidates:
                            for candidate in response.candidates:
                                # Check finish reason if available
                                if hasattr(candidate, 'finish_reason'):
                                    finish_reason = candidate.finish_reason
                                    logger.info(f"Finish reason: {finish_reason}")
                                
                                # Extract safety ratings from candidate
                                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                                    ratings = [f"{r.category.name}:{r.probability.name}" for r in candidate.safety_ratings if hasattr(r, 'category') and hasattr(r, 'probability')]
                                    if ratings:
                                        safety_info = f"Safety ratings: {', '.join(ratings)}. "
                                    
                                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts is not None:
                                    for part in candidate.content.parts:
                                        # Extract text
                                        if hasattr(part, 'text') and part.text:
                                            # Skip accidental HTML payloads
                                            if ("<!DOCTYPE html" in part.text) or ("<html" in part.text.lower()):
                                                pass
                                            else:
                                                response_text += part.text + "\n"

                                        # Extract image data
                                        if hasattr(part, 'inline_data') and part.inline_data:
                                            try:
                                                image_binary = part.inline_data.data
                                                batch_images_bytes.append(image_binary)
                                            except Exception as img_error:
                                                logger.error(
                                                    f"Error extracting image from response: {str(img_error)}"
                                                )

                        # Accumulate images and text (don't process them yet)
                        if batch_images_bytes:
                            all_generated_images_bytes.extend(batch_images_bytes)
                            status_text += (
                                f"Batch {i+1} (seed {current_seed}): Generated {len(batch_images_bytes)} images\n"
                            )
                            if response_text.strip():
                                all_generated_text.append(f"Batch {i+1}:\n{response_text.strip()}")
                        else:
                            # Include finish reason in status if available
                            finish_info = f" (Reason: {finish_reason})" if finish_reason else ""
                            status_text += f"Batch {i+1} (seed {current_seed}): No images found in response{finish_info}\n"
                            
                            # Include additional diagnostic info
                            if prompt_feedback_info:
                                status_text += f"Prompt feedback: {prompt_feedback_info}\n"
                            if safety_info:
                                status_text += f"{safety_info}\n"
                                
                            # Add more specific guidance for IMAGE_SAFETY or similar issues
                            if finish_reason and "SAFETY" in str(finish_reason).upper():
                                status_text += "The request was blocked for safety reasons. Try modifying your prompt to avoid content that might trigger safety filters.\n"
                                
                            # Include any text response from the model that might explain the issue
                            if response_text.strip():
                                status_text += f"Model message: {response_text.strip()}\n"
                                all_generated_text.append(f"Batch {i+1} (no image):\n{response_text.strip()}")

                    except Exception as batch_error:
                        status_text += f"Batch {i+1} error: {str(batch_error)}\n"

            # Process all accumulated images into tensors for ComfyUI only after all steps/batches are complete
            if all_generated_images_bytes:
                logger.info(f"Processing {len(all_generated_images_bytes)} accumulated images")
                
                try:
                    # Convert bytes to PIL images
                    pil_images = []
                    for img_bytes in all_generated_images_bytes:
                        try:
                            pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
                            pil_images.append(pil_img)
                        except Exception as img_error:
                            logger.error(f"Error converting image bytes to PIL: {str(img_error)}")
                    
                    if not pil_images:
                        raise ValueError("Failed to convert any image bytes to PIL images")
                    
                    # Ensure all images have the same dimensions
                    first_width, first_height = pil_images[0].size
                    for i in range(1, len(pil_images)):
                        if pil_images[i].size != (first_width, first_height):
                            pil_images[i] = pil_images[i].resize((first_width, first_height), Image.LANCZOS)
                    
                    # Convert PIL images to tensor format
                    tensors = []
                    for pil_img in pil_images:
                        img_array = np.array(pil_img).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_array)[None,]  # Add batch dimension
                        tensors.append(img_tensor)
                    
                    # Concatenate all image tensors into one batch
                    image_tensors = torch.cat(tensors, dim=0)
                    
                    # Get the actual resolution for reporting
                    height, width = image_tensors.shape[1:3]
                    resolution_info = f"Resolution: {width}x{height}"
                    
                    # Format the result text
                    if sequential_generation:
                        result_text = f"Successfully generated {len(all_generated_images_bytes)} sequential images using {model_name}.\n"
                        result_text += f"Initial prompt: {prompt}\n"
                        result_text += f"Starting seed: {seed}\n"
                        if resolution_info:
                            result_text += f"{resolution_info}\n"
                        
                        # Add text for each step
                        if all_generated_text:
                            result_text += "\n----- Generated Sequence -----\n"
                            result_text += "\n\n".join(all_generated_text)
                    else:
                        result_text = f"Successfully generated {len(all_generated_images_bytes)} images using {model_name}.\n"
                        result_text += f"Prompt: {prompt}\n"
                        result_text += f"Starting seed: {seed}\n"
                        if resolution_info:
                            result_text += f"{resolution_info}\n"
                        
                        # Add text for each batch
                        if all_generated_text:
                            result_text += "\n----- Generated Text -----\n"
                            result_text += "\n\n".join(all_generated_text)
                    
                    # Add status information at the end
                    result_text += f"\n\n----- Generation Status -----\n{status_text}"
                    
                    # Return the final accumulated results
                    return result_text, image_tensors
                    
                except Exception as processing_error:
                    error_msg = f"Error processing accumulated images: {str(processing_error)}"
                    logger.error(error_msg, exc_info=True)
                    return (f"{error_msg}\n\nRaw status:\n{status_text}", create_placeholder_image())
            else:
                # No images were generated successfully
                return (
                    f"No images were generated with {model_name}. Details:\n{status_text}",
                    create_placeholder_image(),
                )

        except Exception as e:
            error_msg = f"Error in image generation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Truncate very long error messages to prevent log spam
            original_error = str(e)
            if len(original_error) > 500:
                error_msg = f"Error in image generation: {original_error[:500]}... [truncated]"
            
            return error_msg, create_placeholder_image()

def get_available_models(api_key):
    """Get available Gemini models for a given API key"""
    try:
        from google import genai
        
        # Initialize client with the provided API key
        raw_client, client_type = create_appropriate_client(api_key, "gemini")
        client = UniversalClient(raw_client, client_type)
        
        # List available models
        models_response = client.models().list()
        
        # Filter for Gemini models only
        gemini_models = []
        for model in models_response:
            model_name_lower = model.name.lower()
            if "gemini" in model_name_lower:
                # Extract just the model name from the full path
                model_name = model.name.split('/')[-1]
                gemini_models.append(model_name)
        
        # Ensure we always have the default models available
        default_models = gemini_models
        
        for model in default_models:
            if model not in gemini_models:
                gemini_models.append(model)
        
        return gemini_models
        
    except Exception as e:
        error_msg = str(e)
        
        # Truncate long error messages
        if len(error_msg) > 300:
            error_msg = error_msg[:300] + "... [truncated]"
        logger.error(f"Error retrieving models: {error_msg}")
        
        # Return default Gemini models
        return gemini_models

def check_gemini_api_key(api_key):
    """Check if a Gemini API key is valid by attempting to list models"""
    try:
        from google import genai
        
        # Initialize client with the provided API key
        raw_client, client_type = create_appropriate_client(api_key, "gemini")
        client = UniversalClient(raw_client, client_type)
        
        # Try to list models as a simple API test
        models = client.models().list()
        
        # If we get here, the API key is valid
        return True, "API key is valid. Successfully connected to Gemini API."
    except Exception as e:
        error_msg = str(e)
        logger.error(f"API key validation error: {error_msg}")
        
        # Truncate very long error messages to prevent log spam
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "... [truncated]"
        
        # Provide more helpful error messages
        if "invalid api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return False, "Invalid API key. Please check your Gemini API key and try again."
        elif "quota" in error_msg.lower() or "exceeded" in error_msg.lower():
            return False, "API quota exceeded. You've reached your usage limit for the Gemini API."
        else:
            return False, f"Error validating API key: {error_msg}"
