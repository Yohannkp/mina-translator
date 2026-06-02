"""
Script de test pour les composants du projet
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# Optionnel: torch peut ne pas etre installe
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch non installe - tests GPU ignores")


def test_settings():
    """Test le module de configuration"""
    logger.info("Test: Configuration...")
    try:
        from config.settings import get_settings
        settings = get_settings()
        logger.info(f"  DATA_DIR: {settings.DATA_DIR}")
        logger.info(f"  WHISPER_MODEL: {settings.WHISPER_MODEL}")
        logger.info("  [OK] Configuration")
        return True
    except Exception as e:
        logger.error(f"  [ERREUR] {e}")
        return False


def test_vocabulary():
    """Test le vocabulaire Mina"""
    logger.info("Test: Vocabulaire Mina...")
    try:
        from data.mina_vocabulary import get_corpus_data, get_audio_phrases

        corpus = get_corpus_data()
        audio_phrases = get_audio_phrases()

        logger.info(f"  Phrases corpus: {len(corpus)}")
        logger.info(f"  Phrases audio: {len(audio_phrases)}")

        logger.info("\n  Exemples:")
        for item in corpus[:5]:
            logger.info(f"    MINA: {item['text']:15} FR: {item['translation']}")

        logger.info("  [OK] Vocabulaire")
        return True
    except Exception as e:
        logger.error(f"  [ERREUR] {e}")
        return False


def test_gpu():
    """Test la détection GPU"""
    logger.info("Test: GPU Detection...")
    try:
        if not TORCH_AVAILABLE:
            logger.warning("  [ATTENTION] PyTorch non installe")
            return True

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"  GPU: {gpu_name}")
            logger.info(f"  VRAM: {vram:.1f} GB")
            logger.info("  [OK] GPU detecte")
        else:
            logger.warning("  [ATTENTION] Aucun GPU CUDA detecte")
            logger.info("  Mode CPU uniquement")
        return True
    except Exception as e:
        logger.error(f"  [ERREUR] {e}")
        return False


def test_corpus_manager():
    """Test le corpus manager"""
    logger.info("Test: Corpus Manager...")
    try:
        from data.corpus_manager import CorpusManager

        manager = CorpusManager()
        stats = manager.get_stats()

        logger.info(f"  Pending: {stats.pending}")
        logger.info(f"  Total texts: {stats.total_texts}")

        logger.info("  [OK] Corpus Manager")
        return True
    except Exception as e:
        logger.error(f"  [ERREUR] {e}")
        return False


def test_stt_pipeline():
    """Test le pipeline STT (sans téléchargement de modèle)"""
    logger.info("Test: STT Pipeline (structure)...")
    try:
        from ml.stt.pipeline import STTPipeline

        # Juste vérifier que la classe existe
        logger.info(f"  Classe STTPipeline: OK")

        # Vérifier les imports
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        logger.info("  Transformers imports: OK")
        logger.info("  [OK] STT Pipeline pret")
        return True
    except Exception as e:
        logger.error(f"  [ERREUR] {e}")
        return False


def test_translation_pipeline():
    """Test le pipeline traduction (sans téléchargement de modèle)"""
    logger.info("Test: Translation Pipeline (structure)...")
    try:
        from ml.translation.pipeline import TranslationPipeline

        logger.info(f"  Classe TranslationPipeline: OK")

        # Vérifier les imports
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("  Transformers imports: OK")
        logger.info("  [OK] Translation Pipeline pret")
        return True
    except Exception as e:
        logger.error(f"  [ERREUR] {e}")
        return False


def test_fastapi():
    """Test l'API FastAPI (sans démarrage)"""
    logger.info("Test: FastAPI (structure)...")
    try:
        from api.main import app

        logger.info(f"  App name: {app.title}")
        logger.info(f"  Version: {app.version}")

        routes = [route.path for route in app.routes]
        logger.info(f"  Routes: {len(routes)}")

        logger.info("  [OK] FastAPI pret")
        return True
    except Exception as e:
        logger.error(f"  [ERREUR] {e}")
        return False


def test_augmentation():
    """Test le module d'augmentation"""
    logger.info("Test: Data Augmentation...")
    try:
        from data.augmentation.augmenter import TextAugmenter, AudioAugmenter

        text_aug = TextAugmenter()

        # Tester une transformation
        test_text = "Mɛlɔ nyabi"
        result = text_aug.random_insertion(test_text, n=1)

        logger.info(f"  Test insertion: '{test_text}' -> '{result}'")
        logger.info("  [OK] Augmentation")
        return True
    except Exception as e:
        logger.error(f"  [ERREUR] {e}")
        return False


def main():
    """Lance tous les tests"""
    print("\n" + "=" * 60)
    print("TESTS DU PROJET MINA-FRANCAIS")
    print("=" * 60 + "\n")

    tests = [
        ("Configuration", test_settings),
        ("Vocabulaire Mina", test_vocabulary),
        ("GPU Detection", test_gpu),
        ("Corpus Manager", test_corpus_manager),
        ("STT Pipeline", test_stt_pipeline),
        ("Translation Pipeline", test_translation_pipeline),
        ("FastAPI", test_fastapi),
        ("Data Augmentation", test_augmentation),
    ]

    results = []
    for name, test_func in tests:
        print()
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test {name} a echoue: {e}")
            results.append((name, False))

    # Résumé
    print("\n" + "=" * 60)
    print("RESUME DES TESTS")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[OK]" if result else "[ECHEC]"
        print(f"  {status:8} {name}")

    print("\n" + "-" * 60)
    print(f"  Resultat: {passed}/{total} tests reussis")

    if passed == total:
        print("\n  Tous les composants sont operationnels!")
    else:
        print(f"\n  {total - passed} test(s) a/ont echoue(s)")

    print("=" * 60)


if __name__ == "__main__":
    main()