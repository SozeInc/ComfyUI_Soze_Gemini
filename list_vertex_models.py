"""
Diagnostic: list the Gemini/Imagen models your Vertex AI project can actually
access. Run it with the same Python environment ComfyUI uses, e.g.:

    .venv\\Scripts\\python.exe list_vertex_models.py

It reads the same VERTEX_API_KEY / VERTEX_PROJECT_ID / VERTEX_LOCATION config
the nodes use (env vars, shell config, or .env). If a model you want is NOT in
the output, that explains a "404 NOT_FOUND ... was not found or your project
does not have access to it" error -- pick one that is listed, or switch region.
"""

import os
import sys

# Allow running both as `python list_vertex_models.py` (script) and as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ComfyUI_Soze_Gemini.env_utils import get_vertex_config
    from ComfyUI_Soze_Gemini.gemini_node import create_vertex_client
except ImportError:
    # Fallback when run from inside the package directory.
    from env_utils import get_vertex_config
    from gemini_node import create_vertex_client


def main():
    cfg = get_vertex_config()
    api_key = cfg["api_key"]
    project = cfg["project"]
    location = cfg["location"]

    mode = "express API key (-> GLOBAL endpoint)" if api_key else "ADC (project/location)"
    print("=" * 70)
    print(f"Auth mode : {mode}")
    print(f"Project   : {project or '(none)'}")
    print(f"Location  : {location or '(none)'}")
    print("=" * 70)

    if not api_key and not project:
        print("ERROR: No credentials. Set VERTEX_API_KEY or VERTEX_PROJECT_ID + VERTEX_LOCATION.")
        return 1

    client = create_vertex_client(
        api_key=api_key or None,
        project=project or None,
        location=location or None,
        api_key_source="diagnostic",
    )

    # Show the endpoint that will actually be hit -- this is the crux of the
    # "404 in us-central1" problem: the Gemini 3.x image models are GLOBAL only.
    ac = getattr(client, "_api_client", None)
    resolved_loc = getattr(ac, "location", None)
    base_url = getattr(getattr(ac, "_http_options", None), "base_url", None)
    print(f"Resolved location : {resolved_loc or '(global/none)'}")
    print(f"Resolved endpoint : {base_url}")
    if base_url and "aiplatform.googleapis.com/" in base_url and "-aiplatform" not in base_url:
        print("  -> GLOBAL endpoint. Good for gemini-3.x image models.")
    else:
        print("  -> REGIONAL endpoint. gemini-3.x image models will 404 here; "
              "set VERTEX_LOCATION=global (ADC) or use an express key.")
    print("=" * 70)

    print("\nModels visible to this project/region:\n")
    found = 0
    try:
        for model in client.models.list():
            name = getattr(model, "name", "") or ""
            short = name.split("/")[-1]
            actions = (
                getattr(model, "supported_actions", None)
                or getattr(model, "supported_generation_methods", None)
                or []
            )
            is_image = "image" in short.lower()
            flag = "  <-- image-capable" if is_image else ""
            extra = f"  actions={list(actions)}" if actions else ""
            print(f"  {short}{extra}{flag}")
            found += 1
    except Exception as e:
        print(f"\nmodels.list() failed: {e}")
        print(
            "\nNote: express-mode keys sometimes cannot enumerate the catalog. "
            "In that case, try the model IDs from the Vertex docs directly and "
            "watch for 404s, or switch to ADC auth."
        )
        return 2

    print(f"\nTotal: {found} model(s).")
    if found == 0:
        print(
            "No models returned. This usually means the catalog isn't enumerable "
            "for your auth mode -- try the documented model IDs directly."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
