"""Cuenta tokens (tokenizer real de qwen35-9b, vía el endpoint /tokenize de vLLM)
sobre todos los .md de salida_md/.

Uso:
    python contar_tokens_md.py
    python contar_tokens_md.py --out ../../datos/tokens_md.json --md-dir /otra/ruta/salida_md

Requiere el servicio vLLM de qwen35-9b (puerto 8002 por defecto, mismo rol que
GENERATE_LLM_URL/GENERATE_LLM_MODEL) levantado.
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

from rag312.config import get_generate_config, settings


def build_tokenize_client(role_config) -> httpx.Client:
    """Cliente HTTP para /tokenize de vLLM: vive en la raíz del server, no bajo /v1
    (a diferencia de /chat/completions), y no está cubierto por el SDK de openai."""
    base_url = role_config.url.removesuffix("/v1")
    return httpx.Client(base_url=base_url, timeout=30.0)


def contar_tokens(client: httpx.Client, model: str, texto: str) -> int:
    resp = client.post("/tokenize", json={"model": model, "prompt": texto, "add_special_tokens": False})
    resp.raise_for_status()
    return resp.json()["count"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str, default=str(settings.datos_dir / "tokens_md.json"))
    parser.add_argument("--md-dir", type=str, default=str(settings.salida_md_dir))
    args = parser.parse_args()

    md_dir = Path(args.md_dir)
    out_path = Path(args.out)

    if not md_dir.exists():
        print(f"ERROR: no existe {md_dir}. Ajustá RAG_PROJECT_DIR o --md-dir.")
        sys.exit(1)

    role_config = get_generate_config()
    client = build_tokenize_client(role_config)

    archivos = sorted(md_dir.rglob("*.md"))
    if not archivos:
        print(f"ERROR: no se encontraron .md en {md_dir}.")
        sys.exit(1)

    detalle = []
    por_categoria: dict[str, int] = {}
    total_tokens = 0

    print(f"Modelo: {role_config.model}  ({role_config.url})")
    print(f"Carpeta: {md_dir}  ({len(archivos)} archivos)")
    print("=" * 60)

    for i, md_path in enumerate(archivos, start=1):
        categoria = md_path.parent.name
        texto = md_path.read_text(encoding="utf-8")
        try:
            tokens = contar_tokens(client, role_config.model, texto)
        except httpx.HTTPError as e:
            print(f"ERROR: no se pudo contactar /tokenize en {role_config.url} ({e}).")
            print("Verificá que el servicio vLLM de qwen35-9b (puerto 8002) esté levantado.")
            sys.exit(1)

        archivo_rel = str(md_path.relative_to(md_dir))
        detalle.append({
            "archivo": archivo_rel,
            "categoria": categoria,
            "caracteres": len(texto),
            "tokens": tokens,
        })
        por_categoria[categoria] = por_categoria.get(categoria, 0) + tokens
        total_tokens += tokens
        print(f"[{i}/{len(archivos)}] {archivo_rel}: {tokens} tokens")

    resultado = {
        "modelo": role_config.model,
        "resumen": {
            "total_archivos": len(archivos),
            "total_tokens": total_tokens,
            "por_categoria": por_categoria,
        },
        "detalle": detalle,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("RESUMEN")
    print(f"  Total archivos: {len(archivos)}")
    print(f"  Total tokens:   {total_tokens}")
    print("  Por categoría:")
    for categoria, tokens in sorted(por_categoria.items()):
        print(f"    {categoria}: {tokens}")
    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
