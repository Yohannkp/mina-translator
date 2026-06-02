"""
Script d'installation et de configuration du projet Mina-Français
用法: python setup.py
"""
import os
import sys
import subprocess
from pathlib import Path
from loguru import logger

# Configuration
PROJECT_ROOT = Path(__file__).parent
PYTHON = sys.executable


def print_banner():
    """Affiche la banniere du projet"""
    print("\n" + "=" * 60)
    print("  IA TRADUCTION MINA-FRANCAIS - SETUP")
    print("  Systeme de traduction et transcription Mina-Francais")
    print("=" * 60 + "\n")


def check_python():
    """Verifie la version de Python"""
    version = sys.version_info
    logger.info(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 10):
        logger.error("Python 3.10+ requis")
        return False
    return True


def check_cuda():
    """Verifie l'installation CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU detecte: {gpu}")
            logger.info(f"VRAM: {vram:.1f} GB")
            return True
        else:
            logger.warning("CUDA non disponible - mode CPU uniquement")
            return False
    except ImportError:
        logger.warning("PyTorch non installe - impossible de verifier CUDA")
        return False


def install_requirements():
    """Installe les dependances Python"""
    logger.info("Installation des dependances...")

    # Core dependencies
    core_packages = [
        "loguru",
        "pydantic>=2.6.0",
        "pydantic-settings>=2.2.0",
        "python-dotenv>=1.0.0",
    ]

    # ML dependencies
    ml_packages = [
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "huggingface_hub>=0.21.0",
        "accelerate>=0.30.0",
        "peft>=0.10.0",
    ]

    # API dependencies
    api_packages = [
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "python-multipart>=0.0.9",
        "aiofiles>=23.2.1",
    ]

    # Audio dependencies
    audio_packages = [
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "numpy>=1.26.0",
        "scipy>=1.12.0",
    ]

    # Web scraping
    web_packages = [
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
    ]

    all_packages = core_packages + ml_packages + api_packages + audio_packages + web_packages

    for pkg in all_packages:
        try:
            logger.info(f"  Installation {pkg}...")
            subprocess.check_call([PYTHON, "-m", "pip", "install", pkg, "--quiet"])
        except Exception as e:
            logger.warning(f"  Erreur installation {pkg}: {e}")

    logger.info("Installation terminee!")


def create_directories():
    """Cree les repertoires necessaires"""
    logger.info("Creation des repertoires...")

    directories = [
        PROJECT_ROOT / "data" / "corpus" / "raw",
        PROJECT_ROOT / "data" / "corpus" / "processed",
        PROJECT_ROOT / "data" / "corpus" / "audio",
        PROJECT_ROOT / "data" / "corpus" / "annotations",
        PROJECT_ROOT / "data" / "corpus" / "export",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "logs",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"  {directory}")

    logger.info("Repertoires crees!")


def init_corpus():
    """Initialise le corpus avec les donnees de base"""
    logger.info("Initialisation du corpus Mina...")

    try:
        # Import et execution
        sys.path.insert(0, str(PROJECT_ROOT))
        from data.corpus_manager import CorpusManager
        from data.mina_vocabulary import get_corpus_data

        manager = CorpusManager()

        # Ajouter le vocabulaire de base
        corpus_data = get_corpus_data()
        manager.add_manual_entries(corpus_data)

        logger.info(f"  {len(corpus_data)} phrases ajoutees")

        # Afficher les statistiques
        stats = manager.get_stats()
        logger.info(f"  Total pending: {stats.pending}")

        logger.info("Corpus initialise!")

    except Exception as e:
        logger.warning(f"Erreur initialisation corpus: {e}")
        logger.info("Le corpus pourra etre initialise manuellement plus tard")


def create_env_file():
    """Cree le fichier .env example"""
    logger.info("Creation du fichier .env...")

    env_content = """# IA Traduction Mina-Francais - Configuration

# API
API_HOST=0.0.0.0
API_PORT=8000

# Modeles
WHISPER_MODEL=openai/whisper-small
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# GPU
MAX_VRAM_USAGE_GB=7.5
BATCH_SIZE=4

# Cache
CACHE_TTL_SECONDS=3600

# Logging
LOG_LEVEL=INFO
"""

    env_file = PROJECT_ROOT / ".env.example"
    with open(env_file, "w", encoding="utf-8") as f:
        f.write(env_content)

    logger.info(f"  Fichier .env.example cree")


def create_dockerfile():
    """Cree un Dockerfile pour le déploiement"""
    logger.info("Creation du Dockerfile...")

    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Installer CUDA (pour GPU)
RUN apt-get update && apt-get install -y --no-install-recommends \\
    cuda-cudart-12-1 \\
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Exposer le port
EXPOSE 8000

# Commande de demarrage
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

    dockerfile = PROJECT_ROOT / "Dockerfile"
    with open(dockerfile, "w", encoding="utf-8") as f:
        f.write(dockerfile_content)

    logger.info("  Dockerfile cree")


def create_docker_compose():
    """Cree un docker-compose.yml"""
    logger.info("Creation du docker-compose.yml...")

    compose_content = """version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
"""

    compose_file = PROJECT_ROOT / "docker-compose.yml"
    with open(compose_file, "w", encoding="utf-8") as f:
        f.write(compose_content)

    logger.info("  docker-compose.yml cree")


def create_gitignore():
    """Met a jour .gitignore"""
    logger.info("Mise a jour .gitignore...")

    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
*.so
.Python
env/
venv/
.venv/

# Logs
logs/*.log

# Data
data/corpus/raw/*
data/corpus/audio/*
!data/corpus/raw/.gitkeep
!data/corpus/audio/.gitkeep

# Models
models/*
!models/.gitkeep

# Cache
*.cache
.cache/

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo
"""

    gitignore_file = PROJECT_ROOT / ".gitignore"
    with open(gitignore_file, "w", encoding="utf-8") as f:
        f.write(gitignore_content)

    logger.info("  .gitignore mis a jour")


def main():
    """Point d'entree principal"""
    print_banner()

    # Verifier Python
    if not check_python():
        sys.exit(1)

    # Creer les repertoires
    create_directories()

    # Creer les fichiers de config
    create_env_file()
    create_dockerfile()
    create_docker_compose()
    create_gitignore()

    # Initialiser le corpus
    init_corpus()

    print("\n" + "=" * 60)
    print("SETUP TERMINE!")
    print("=" * 60)
    print("\nProchaines etapes:")
    print("  1. Creer environnement virtuel (recommande):")
    print("     python -m venv venv")
    print("     .\\venv\\Scripts\\activate")
    print()
    print("  2. Installer les dependances:")
    print("     pip install -r requirements.txt")
    print()
    print("  3. Pour PyTorch GPU (RTX 4060):")
    print("     pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print()
    print("  4. Lancer l'API:")
    print("     python -m api.main")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()