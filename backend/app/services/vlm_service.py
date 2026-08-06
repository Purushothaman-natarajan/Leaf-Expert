"""
Leaf-Expert — VLM Service
Multi-provider vision-language model adapter for zero-shot leaf disease analysis.

Supported providers:
  • Gemini 2.0 Flash  (google-genai)     — default, native JSON schema
  • OpenAI GPT-4o     (openai)           — Structured Outputs
  • Claude 3.5 Sonnet (anthropic)        — XML-guided JSON
  • Ollama LLaVA      (httpx, local)     — offline / private

API keys are NEVER stored server-side; passed per-request via X-VLM-API-Key header.
"""
from __future__ import annotations

import base64
import json
import time
import uuid
from io import BytesIO
from typing import Optional

from PIL import Image

from app.core.logging import get_logger
from app.schemas.vlm import LeafScanResult, VLMProvider

logger = get_logger(__name__)

# ─── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert botanist and plant pathologist with 20+ years of field experience.

Analyze the provided leaf image with clinical precision and return a structured JSON diagnosis.

STRICT RULES:
1. If the image does NOT show a plant leaf, set is_plant=false, is_diseased=false, disease_name="Not a plant".
2. Be conservative with confidence — only set confidence > 0.85 if you are highly certain.
3. disease_name must be specific (e.g., "Tomato Late Blight", not just "blight"). Use EPPO codes when possible.
4. If the leaf appears healthy, set is_diseased=false, disease_name="Healthy", severity="none".
5. affected_area_percent: estimate the % of the visible leaf surface showing disease symptoms (0 if healthy).
6. treatment: be specific — include product types/active ingredients, not brand names. Max 6 items.
7. symptoms: list only what you can visibly observe. Max 6 items.
8. urgency: base on rate of disease spread and economic/plant impact.
9. Return ONLY the JSON object. No markdown fences, no preamble, no commentary.
"""

USER_PROMPT = (
    "Analyze this leaf image and return a JSON diagnosis following your instructions exactly."
)

# ─── Provider defaults ──────────────────────────────────────────────────────────

PROVIDER_DEFAULTS = {
    VLMProvider.gemini: {
        "model": "gemini-2.0-flash",
        "name": "Google Gemini 2.0 Flash",
        "cost_tier": "low",
        "requires_api_key": True,
        "description": "Recommended — fast, cheap, native JSON schema enforcement",
    },
    VLMProvider.openai: {
        "model": "gpt-4o",
        "name": "OpenAI GPT-4o",
        "cost_tier": "medium",
        "requires_api_key": True,
        "description": "Most capable, higher cost per image",
    },
    VLMProvider.claude: {
        "model": "claude-3-5-sonnet-20241022",
        "name": "Anthropic Claude 3.5 Sonnet",
        "cost_tier": "medium",
        "requires_api_key": True,
        "description": "Excellent reasoning and detailed explanations",
    },
    VLMProvider.ollama: {
        "model": "llava:latest",
        "name": "Ollama (Local)",
        "cost_tier": "free",
        "requires_api_key": False,
        "description": "Fully local/offline — no API key needed, privacy-first",
    },
}


# ─── Image helpers ──────────────────────────────────────────────────────────────

def _pil_to_base64(img: Image.Image, max_size: int = 1024) -> str:
    """Resize and base64-encode image for APIs that need it (OpenAI, Claude, Ollama)."""
    img = img.convert("RGB")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_json_result(raw: str) -> LeafScanResult:
    """Parse raw string output into LeafScanResult; strip markdown fences if present."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    data = json.loads(raw)
    return LeafScanResult(**data)


# ─── Gemini provider ────────────────────────────────────────────────────────────

def _scan_gemini(image: Image.Image, api_key: str, model: str) -> LeafScanResult:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("google-genai package not installed. Run: pip install google-genai")

    client = genai.Client(api_key=api_key)
    img = image.convert("RGB")
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024), Image.LANCZOS)

    response = client.models.generate_content(
        model=model,
        contents=[img, USER_PROMPT],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=LeafScanResult,
            temperature=0.1,  # low temperature for deterministic diagnosis
        ),
    )
    return response.parsed


# ─── OpenAI provider ────────────────────────────────────────────────────────────

def _scan_openai(image: Image.Image, api_key: str, model: str) -> LeafScanResult:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key)
    b64 = _pil_to_base64(image)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=1500,
    )
    raw = response.choices[0].message.content or ""
    return _parse_json_result(raw)


# ─── Claude provider ────────────────────────────────────────────────────────────

def _scan_claude(image: Image.Image, api_key: str, model: str) -> LeafScanResult:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    client = anthropic.Anthropic(api_key=api_key)
    b64 = _pil_to_base64(image)

    message = client.messages.create(
        model=model,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            }
        ],
        temperature=0.1,
    )
    raw = message.content[0].text if message.content else ""
    return _parse_json_result(raw)


# ─── Ollama provider (local) ────────────────────────────────────────────────────

def _scan_ollama(
    image: Image.Image,
    model: str,
    ollama_host: str = "http://localhost:11434",
) -> LeafScanResult:
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx not installed. Run: pip install httpx")

    b64 = _pil_to_base64(image)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}\n\nRespond with ONLY valid JSON."

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "images": [b64],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1},
            },
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")

    return _parse_json_result(raw)


# ─── Public API ─────────────────────────────────────────────────────────────────

def scan_leaf(
    image_path: str,
    provider: VLMProvider,
    api_key: Optional[str],
    model_name: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
) -> tuple[str, LeafScanResult, str, str, int]:
    """
    Analyze a leaf image using the specified VLM provider.

    Returns:
        (scan_id, result, provider_str, model_str, processing_ms)
    """
    defaults = PROVIDER_DEFAULTS[provider]
    model = model_name or defaults["model"]

    if defaults["requires_api_key"] and not api_key:
        raise ValueError(f"API key required for provider '{provider.value}'")

    image = Image.open(image_path).convert("RGB")
    t0 = time.time()

    logger.info(f"Scanning with {provider.value} / {model}")
    try:
        if provider == VLMProvider.gemini:
            result = _scan_gemini(image, api_key, model)  # type: ignore[arg-type]
        elif provider == VLMProvider.openai:
            result = _scan_openai(image, api_key, model)  # type: ignore[arg-type]
        elif provider == VLMProvider.claude:
            result = _scan_claude(image, api_key, model)  # type: ignore[arg-type]
        elif provider == VLMProvider.ollama:
            result = _scan_ollama(image, model, ollama_host)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    except (json.JSONDecodeError, Exception) as e:
        logger.exception(f"VLM scan failed: {e}")
        raise

    elapsed_ms = int((time.time() - t0) * 1000)
    scan_id = uuid.uuid4().hex
    logger.info(
        f"Scan complete — {provider.value}/{model} | "
        f"disease={result.disease_name} | conf={result.confidence:.2f} | {elapsed_ms}ms"
    )
    return scan_id, result, provider.value, model, elapsed_ms


def get_providers_info() -> list[dict]:
    return [
        {
            "id": p.value,
            "name": info["name"],
            "default_model": info["model"],
            "cost_tier": info["cost_tier"],
            "requires_api_key": info["requires_api_key"],
            "description": info["description"],
        }
        for p, info in PROVIDER_DEFAULTS.items()
    ]
