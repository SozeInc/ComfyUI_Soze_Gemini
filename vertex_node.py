"""
Vertex AI node for ComfyUI (Soze).

Talks to Google Cloud's Vertex AI rather than AI Studio, using the same
google-genai SDK as gemini_node.py -- only the client construction differs
(vertexai=True). Credentials are read from a dedicated set of environment
variables (VERTEX_API_KEY / VERTEX_PROJECT_ID / VERTEX_LOCATION) so they never
collide with the AI Studio GEMINI_API_KEY.

  * Soze_VertexGemini - Gemini text / multimodal / image generation via Vertex.
                        Subclasses Soze_GeminiAdvanced and reuses all of its
                        generation logic, overriding only the backend seams.
"""

import logging

from .gemini_node import (
    Soze_GeminiAdvanced,
    create_vertex_client,
)
from .env_utils import get_vertex_config

logger = logging.getLogger(__name__)

# Sentinel returned by _resolve_api_key when no express key is configured but a
# project is, meaning "fall back to Application Default Credentials".
_ADC_SENTINEL = "__VERTEX_ADC__"

# Gemini model IDs as published on Vertex AI. NOTE: these differ from the AI
# Studio names, and preview IDs get retired without the usual notice period.
# Exact availability also depends on your project, region, allowlisting, and
# whether you are on express mode -- run list_vertex_models.py if a model
# returns 404 NOT_FOUND. Recently retired IDs (do NOT re-add):
#   gemini-3-pro-preview        -> shut down 2026-03-09 (use gemini-3.1-pro-preview)
#   gemini-3-pro-image-preview  -> shut down 2026-06-25 (use gemini-3-pro-image, GA)
#   gemini-3.1-flash-image-preview -> shut down 2026-06-25 (use gemini-3.1-flash-image)
vertex_gemini_models = [
    # --- Image generation (Nano Banana family) ---
    # The Gemini 3.x image models are GLOBAL-endpoint only -- they 404 in a
    # region like us-central1. Use auth_mode=express (always global) or
    # auth_mode=adc with VERTEX_LOCATION=global. gemini-3.1-flash-image is
    # first so it stays the default / image fallback.
    "gemini-3.1-flash-image",        # Nano Banana 2 (image) - GA; GLOBAL only
    "gemini-3.1-flash-lite-image",   # Nano Banana 2 Lite (image) - GA; GLOBAL only
    "gemini-3-pro-image",            # Nano Banana Pro (image) - GA; GLOBAL only; up to 4K
    "gemini-2.5-flash-image",        # Nano Banana (image) - GA
    # --- Text / multimodal ---
    "gemini-3.1-pro-preview",        # replaces gemini-3-pro-preview; GLOBAL endpoint only
    "gemini-3-flash-preview",        # GLOBAL endpoint only
    "gemini-3.1-flash-lite-preview",
]

# Subset of the above that can return images (first entry is the fallback).
vertex_gemini_image_models = [
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
    "gemini-3-pro-image",
    "gemini-2.5-flash-image",
]


class Soze_VertexGemini(Soze_GeminiAdvanced):
    """Gemini on Vertex AI. Reuses Soze_GeminiAdvanced's generation pipeline."""

    CATEGORY = "Soze Nodes"

    # Vertex publishes image-capable models under different IDs than AI Studio.
    IMAGE_MODELS = vertex_gemini_image_models

    def __init__(self):
        # Deliberately do NOT call super().__init__() -- that loads the AI Studio
        # GEMINI_API_KEY and warns if it is missing, which is irrelevant here.
        from .utils import ChatHistory

        cfg = get_vertex_config()
        self.vertex_api_key = cfg["api_key"]
        self.vertex_project = cfg["project"]
        self.vertex_location = cfg["location"]

        self.chat_history = ChatHistory()
        self.last_external_api_key = ""
        self.api_key = self.vertex_api_key
        self.api_key_source = "vertex"
        self.auth_mode = "auto"
        self.genai_available = self._check_genai_availability()

    @classmethod
    def INPUT_TYPES(s):
        # Start from the AI Studio node's inputs (a fresh dict each call) and
        # swap in the Vertex model list plus the Vertex credential overrides.
        base = Soze_GeminiAdvanced.INPUT_TYPES()
        base["required"]["model_name"] = (
            vertex_gemini_models,
            {"default": vertex_gemini_models[0]},
        )
        # Add a "none" thinking level (model default / no thinking config sent).
        base["required"]["thinking_level"] = (
            ["none", "minimal", "low", "medium", "high"],
            {"default": "minimal"},
        )
        opt = base["optional"]
        # Auth mode lets you keep BOTH VERTEX_API_KEY and VERTEX_PROJECT_ID/
        # VERTEX_LOCATION configured at once and choose per run which to use --
        # the google-genai SDK forbids passing an API key and project/location
        # to the same client (they are mutually exclusive). Notes:
        #   express -> uses VERTEX_API_KEY; region is ignored, always GLOBAL endpoint
        #   adc     -> uses project + location (real region you control) via ADC,
        #              ignoring any express key
        #   auto    -> express key if present, otherwise ADC
        opt["auth_mode"] = (
            ["auto", "express", "adc"],
            {"default": "auto"},
        )
        # external_api_key already exists -- on this node it is the Vertex
        # express-mode API key override.
        opt["vertex_project"] = ("STRING", {"default": ""})
        opt["vertex_location"] = ("STRING", {"default": ""})
        return base

    # --- Backend seams overridden from Soze_GeminiAdvanced -----------------

    def _resolve_api_key(self, external_api_key=""):
        mode = getattr(self, "auth_mode", "auto")
        cleaned = external_api_key.strip() if external_api_key else ""

        # A node-provided express key always wins (unless ADC is forced).
        if cleaned and mode != "adc":
            logger.info("Using external (node-provided) Vertex express API key")
            return cleaned, "vertex-external"

        if mode == "adc":
            # Force ADC: use project/location, ignore any express key.
            if self.vertex_project:
                logger.info("auth_mode=adc: using Application Default Credentials")
                return _ADC_SENTINEL, "vertex-adc"
            return None, None

        if mode == "express":
            # Force express: use the API key, never ADC.
            if self.vertex_api_key:
                return self.vertex_api_key, "vertex-env"
            return None, None

        # auto: express key if we have one, otherwise fall back to ADC.
        if self.vertex_api_key:
            return self.vertex_api_key, "vertex-env"
        if self.vertex_project:
            logger.info("auth_mode=auto: no express key, falling back to ADC")
            return _ADC_SENTINEL, "vertex-adc"
        return None, None

    def _no_key_error(self):
        mode = getattr(self, "auth_mode", "auto")
        if mode == "express":
            return (
                "ERROR: auth_mode=express but no VERTEX_API_KEY (or external_api_key) is set. "
                "Provide an express key, or switch auth_mode to 'adc'/'auto'."
            )
        if mode == "adc":
            return (
                "ERROR: auth_mode=adc but no project is set. Set VERTEX_PROJECT_ID + "
                "VERTEX_LOCATION (or the node's vertex_project/vertex_location) and run "
                "`gcloud auth application-default login`, or switch auth_mode to 'express'/'auto'."
            )
        return (
            "ERROR: No Vertex AI credentials found. Set VERTEX_API_KEY (express mode), "
            "or set VERTEX_PROJECT_ID + VERTEX_LOCATION and authenticate with "
            "`gcloud auth application-default login`. You can also paste an express "
            "key into external_api_key and set project/location on the node."
        )

    def _make_client(self, api_key, api_key_source="vertex"):
        key = None if api_key == _ADC_SENTINEL else api_key
        client = create_vertex_client(
            api_key=key,
            project=self.vertex_project,
            location=self.vertex_location,
            api_key_source=api_key_source,
        )
        # client_type "gemini" -> UniversalClient drives it exactly like the
        # AI Studio client (same models.generate_content interface).
        return client, "gemini"

    # --- Pass node-level credential overrides through to the instance ------

    def generate_content(self, *args, vertex_project="", vertex_location="", auth_mode="auto", **kwargs):
        self.auth_mode = auth_mode or "auto"
        if vertex_project and vertex_project.strip():
            self.vertex_project = vertex_project.strip()
        if vertex_location and vertex_location.strip():
            self.vertex_location = vertex_location.strip()
        return super().generate_content(*args, **kwargs)
