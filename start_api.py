"""
start_api.py - Lance l'API Mina-Translator en production
========================================================

API avec modèle fine-tuné et support FR↔Mina

Usage:
    python start_api.py              # Mode développement (8000)
    python start_api.py --port 8080  # Port personnalisé

Auteur: Claude Opus 4.8
Date: 2026-06-04
"""

import sys
import io
import argparse

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def main():
    parser = argparse.ArgumentParser(description="Lance l'API Mina-Translator")
    parser.add_argument("--port", type=int, default=8000, help="Port de l'API (défaut: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Hôte (défaut: 0.0.0.0)")
    parser.add_argument("--workers", type=int, default=1, help="Nombre de workers (défaut: 1)")
    args = parser.parse_args()

    import uvicorn
    from api.mina_api import app

    print()
    print("=" * 60)
    print("MINA-TRANSLATOR API - PRODUCTION")
    print("=" * 60)
    print(f"Port: {args.port}")
    print(f"Host: {args.host}")
    print(f"Workers: {args.workers}")
    print()
    print("Endpoints:")
    print("  GET  /                    - Statut de l'API")
    print("  GET  /health             - Health check")
    print("  POST /translate          - Traduire FR → Mina ou Mina → FR")
    print("  POST /transcribe         - Audio → Texte (STT)")
    print("  POST /tts                - Texte → Audio (TTS)")
    print()
    print("Docs: http://localhost:{}/docs".format(args.port))
    print("=" * 60)
    print()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()