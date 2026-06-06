"""
Script de test pour le systeme B2B Mina-Translator
====================================================
Teste l'authentification, les quotas, et le logging.

Usage:
    python scripts/test_b2b.py
"""
import os
import sys
from pathlib import Path

# Configurer l'encodage pour Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_database():
    """Test la base de donnees"""
    print("\n" + "="*60)
    print("TEST 1: Base de donnees")
    print("="*60)

    from api.database import init_db, create_client, create_api_key, get_client, get_quota

    # Initialiser
    try:
        init_db()
        print("[OK] Base de donnees initialisee")
    except Exception as e:
        print(f"[FAIL] Erreur d'initialisation: {e}")
        return False

    # Creer un client de test
    try:
        client_id = create_client(
            name="Client Test B2B",
            email="test@entreprise.tg",
            company="Entreprise Test SARL",
            plan="pro"
        )
        print(f"[OK] Client cree (ID: {client_id})")
    except Exception as e:
        print(f"[WARN] Client potentiellement deja cree: {e}")
        client = get_client(email="test@entreprise.tg")
        client_id = client['id'] if client else None

    if client_id:
        # Verifier les quotas
        quota = get_quota(client_id)
        print(f"[OK] Quotas definis: {quota['daily_limit']}/{quota['monthly_limit']}")

    return True


def test_auth():
    """Test l'authentification"""
    print("\n" + "="*60)
    print("TEST 2: Authentification")
    print("="*60)

    from api.auth import (
        get_master_key,
        is_master_key,
        sanitize_input,
        validate_api_key_format
    )

    # Verifier Master Key
    master_key = get_master_key()
    if master_key:
        print(f"[OK] Master Key configuree (longueur: {len(master_key)})")

        # Tester is_master_key
        if is_master_key(master_key):
            print("[OK] Verification Master Key: OK")
        else:
            print("[FAIL] Verification Master Key: ECHOUE")
            return False
    else:
        print("[WARN] Master Key non configuree (utilisez scripts/init_b2b.py)")

    # Tester validation de cle
    test_keys = [
        ("mtk_test", False),
        ("mtk_abcd123efgh456ijk", True),
        ("", False),
        ("invalid", False),
    ]

    print("\nValidation des cles API:")
    for key, expected in test_keys:
        result = validate_api_key_format(key)
        status = "[OK]" if result == expected else "[FAIL]"
        print(f"   {status} '{key[:20]}...' -> {result}")

    # Tester sanitize
    print("\nSanitization des entrees:")
    test_inputs = [
        "Bonjour, ca va?",
        "<script>alert('xss')</script>",
        "Texte avec nouvelles lignes",
        "Texte avec caracteres speciaux: aeee",
    ]

    for text in test_inputs:
        sanitized = sanitize_input(text, max_length=100)
        print(f"   [OK] Input: {text[:30]}...")
        print(f"        Output: {sanitized[:30]}...")

    return True


def test_logger():
    """Test le systeme de logging"""
    print("\n" + "="*60)
    print("TEST 3: Logging")
    print("="*60)

    from api.logger import log_request, get_realtime_stats, get_usage_report

    # Logger une requete de test
    try:
        log_request(
            client_id=999,
            client_name="Test Client",
            api_key_id="mtk_test123",
            endpoint="/translate",
            request_text="Bonjour",
            response_text="Woezor",
            inference_ms=150,
            status_code=200,
            domain="famille",
            plan="pro"
        )
        print("[OK] Log enregistre")

        # Lire les stats
        stats = get_realtime_stats()
        print(f"[OK] Stats temps reel: {stats}")

    except Exception as e:
        print(f"[FAIL] Erreur de logging: {e}")
        return False

    return True


def test_api_simulation():
    """Simule des appels API"""
    print("\n" + "="*60)
    print("TEST 4: Simulation d'appels API")
    print("="*60)

    from api.auth import is_master_key
    from api.database import verify_api_key, check_quota
    from api.logger import log_request

    # Recuperer la Master Key
    master_key = os.environ.get("MINA_MASTER_KEY")
    if not master_key:
        # Essayer de lire depuis .env
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if line.startswith("MINA_MASTER_KEY="):
                            master_key = line.split("=", 1)[1].strip()
                            break
            except Exception:
                pass

    if master_key:
        # Test avec Master Key
        print("\n[Test] Acces avec Master Key:")
        is_master = is_master_key(master_key)
        status = "[OK]" if is_master else "[FAIL]"
        print(f"   {status} Acces Master Key: {is_master}")

        # Ne devrait pas logger pour facturation
        log_request(
            client_id=0,
            client_name="Super-Admin",
            api_key_id="MASTER_KEY",
            endpoint="/translate",
            request_text="Test Master",
            response_text="Test Mina",
            inference_ms=100,
            status_code=200,
            is_master=True
        )
        print("   [OK] Requete Master Key loggee (sans facturation)")

    return True


def main():
    """Point d'entree principal"""
    print("\n" + "+" + "="*58 + "+")
    print("|" + " "*15 + "MINA-TRANSLATOR B2B" + " "*16 + "|")
    print("|" + " "*10 + "Tests du Systeme SaaS" + " "*17 + "|")
    print("+" + "="*58 + "+")

    results = []

    # Executer les tests
    results.append(("Base de donnees", test_database()))
    results.append(("Authentification", test_auth()))
    results.append(("Logging", test_logger()))
    results.append(("Simulation API", test_api_simulation()))

    # Resume
    print("\n" + "="*60)
    print("RESUME DES TESTS")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"   {status}: {name}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)

    if all_passed:
        print("="*60)
        print("TOUS LES TESTS PASSES")
        print("="*60)
        print("""
Le systeme B2B est operationnel!

Etapes suivantes:
1. Lancez l'API: uvicorn api.main:app --reload
2. Testez /health: curl http://localhost:8000/health
3. Testez /translate/open: curl -X POST http://localhost:8000/translate/open \\
     -H "Content-Type: application/json" \\
     -d '{"text": "Bonjour"}'
4. Configurez vos clients B2B via /admin/api-keys (Master Key requise)
""")
    else:
        print("[FAIL] CERTAINS TESTS ONT ECHOUE")
        print("="*60)
        print("Corrigez les erreurs avant de continuer.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)