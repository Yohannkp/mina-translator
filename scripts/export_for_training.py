"""
scripts/export_for_training.py - Exporte les traductions au format JSONL
====================================================================

Utilise les traductions crowdsourcées pour créer un fichier d'entraînement.

Usage:
    python scripts/export_for_training.py
"""

import sqlite3
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "mina_crowdsource.db"
OUTPUT_FILE = PROJECT_ROOT / "data" / "corpus" / "crowdsourced_translations.jsonl"


def export_for_training():
    """Exporte les traductions au format JSONL pour l'entraînement"""

    print("=" * 60)
    print("📤 EXPORT POUR L'ENTRAÎNEMENT")
    print("=" * 60)

    if not DB_PATH.exists():
        print("❌ Base non trouvée!")
        return

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Récupérer les traductions
    cursor.execute("""
        SELECT mina_text, french_text
        FROM translations
        WHERE french_text IS NOT NULL
        AND french_text != ''
        AND LENGTH(french_text) > 1
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()

    if not rows:
        print("\n❌ Aucune traduction à exporter!")
        return

    print(f"\n📊 Trouvé {len(rows)} traductions")

    # Créer le dossier de sortie
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Exporter en JSONL (format pour instruction tuning)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for mina, french in rows:
            # Format instruction tuning
            entry = {
                "instruction": "Traduis cette phrase de l'Ewe (Mina du Togo) en Français.",
                "input": mina,
                "output": french
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n✅ Exporté vers: {OUTPUT_FILE}")

    # Afficher un exemple
    print("\n📝 Exemple de format:")
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        example = json.loads(first_line)
        print(f"\n   Input:  {example['input']}")
        print(f"   Output: {example['output']}")

    # Statistiques
    print("\n📈 Statistiques:")
    print(f"   • Total entrées: {len(rows)}")
    print(f"   • Fichier: {OUTPUT_FILE}")
    print(f"   • Taille: {OUTPUT_FILE.stat().st_size} octets")

    conn.close()

    print("\n" + "=" * 60)
    print("💡 Prochaine étape:")
    print("   python scripts/train_mina_llm.py --corpus data/corpus/crowdsourced_translations.jsonl")
    print("=" * 60)


if __name__ == "__main__":
    export_for_training()