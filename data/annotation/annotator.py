"""
Outil d'annotation pour corpus Mina-Français
Interface CLI interactive pour annoter et valider les données
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import random
from loguru import logger

# Configuration
from config.settings import get_settings
settings = get_settings()


@dataclass
class AnnotationEntry:
    """Entrée annotée"""
    id: str
    text: str
    lang: str
    translation: Optional[str] = None
    quality: Optional[str] = None  # "good", "medium", "bad"
    validated: bool = False
    annotator: Optional[str] = None
    notes: Optional[str] = None
    timestamp: Optional[str] = None
    source: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AnnotationEntry":
        return cls(**data)


class AnnotationManager:
    """Gestionnaire d'annotations pour le corpus Mina"""

    def __init__(
        self,
        corpus_dir: Optional[Path] = None,
        annotator_name: str = "anonymous",
    ):
        self.corpus_dir = corpus_dir or settings.CORPUS_DIR
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.annotator_name = annotator_name

        # Fichiers de données
        self.pending_file = self.corpus_dir / "annotation_pending.json"
        self.validated_file = self.corpus_dir / "annotation_validated.json"
        self.rejected_file = self.corpus_dir / "annotation_rejected.json"

        # Charger les données existantes
        self.pending = self._load_json(self.pending_file)
        self.validated = self._load_json(self.validated_file)
        self.rejected = self._load_json(self.rejected_file)

        # Stats
        self.stats = {
            "pending": len(self.pending),
            "validated": len(self.validated),
            "rejected": len(self.rejected),
        }

    def _load_json(self, filepath: Path) -> List[Dict]:
        """Charge un fichier JSON"""
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_json(self, filepath: Path, data: List[Dict]):
        """Sauvegarde un fichier JSON"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _generate_id(self) -> str:
        """Génère un ID unique"""
        return f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"

    def add_entry(
        self,
        text: str,
        lang: str = "mina",
        translation: str = None,
        source: str = None,
    ) -> AnnotationEntry:
        """Ajoute une entrée à annoter"""
        entry = AnnotationEntry(
            id=self._generate_id(),
            text=text,
            lang=lang,
            translation=translation,
            source=source,
            timestamp=datetime.now().isoformat(),
        )
        self.pending.append(entry.to_dict())
        self._save_json(self.pending_file, self.pending)
        logger.info(f"Entry ajoutée: {entry.id}")
        return entry

    def add_batch(self, entries: List[Dict]):
        """Ajoute plusieurs entrées"""
        for entry in entries:
            self.add_entry(
                text=entry.get("text", ""),
                lang=entry.get("lang", "mina"),
                translation=entry.get("translation"),
                source=entry.get("source"),
            )
        self.stats["pending"] = len(self.pending)

    def get_next(self) -> Optional[AnnotationEntry]:
        """Récupère la prochaine entrée à annoter"""
        if self.pending:
            return AnnotationEntry.from_dict(self.pending[0])
        return None

    def validate(
        self,
        entry_id: str,
        translation: str = None,
        quality: str = "good",
        notes: str = None,
    ):
        """Valide une entrée"""
        # Trouver dans pending
        entry_data = None
        for i, e in enumerate(self.pending):
            if e["id"] == entry_id:
                entry_data = e
                self.pending.pop(i)
                break

        if not entry_data:
            logger.warning(f"Entry non trouvée: {entry_id}")
            return False

        # Mettre à jour
        entry_data["validated"] = True
        entry_data["quality"] = quality
        entry_data["notes"] = notes
        entry_data["annotator"] = self.annotator_name

        if translation:
            entry_data["translation"] = translation

        # Déplacer vers validés
        self.validated.append(entry_data)
        self._save_json(self.pending_file, self.pending)
        self._save_json(self.validated_file, self.validated)

        self.stats["pending"] = len(self.pending)
        self.stats["validated"] = len(self.validated)

        return True

    def reject(self, entry_id: str, reason: str = None):
        """Rejette une entrée"""
        entry_data = None
        for i, e in enumerate(self.pending):
            if e["id"] == entry_id:
                entry_data = e
                self.pending.pop(i)
                break

        if not entry_data:
            return False

        entry_data["validated"] = False
        entry_data["notes"] = reason
        entry_data["annotator"] = self.annotator_name

        self.rejected.append(entry_data)
        self._save_json(self.pending_file, self.pending)
        self._save_json(self.rejected_file, self.rejected)

        self.stats["pending"] = len(self.pending)
        self.stats["rejected"] = len(self.rejected)

        return True

    def skip(self, entry_id: str) -> bool:
        """Passe une entrée (la met en fin de liste)"""
        for i, e in enumerate(self.pending):
            if e["id"] == entry_id:
                entry = self.pending.pop(i)
                self.pending.append(entry)
                self._save_json(self.pending_file, self.pending)
                return True
        return False

    def export_parallel_corpus(
        self,
        output_file: Path = None,
        min_quality: str = "medium",
    ) -> Path:
        """
        Exporte un corpus parallèle FR-Mina pour entraînement

        Args:
            output_file: Fichier de sortie
            min_quality: Qualité minimum ("good", "medium")

        Returns:
            Chemin du fichier exporté
        """
        if output_file is None:
            output_file = self.corpus_dir / "parallel_corpus.jsonl"

        quality_order = {"good": 3, "medium": 2, "bad": 1}
        min_level = quality_order.get(min_quality, 2)

        parallel_data = []
        for entry in self.validated:
            quality = entry.get("quality", "medium")
            if quality_order.get(quality, 0) >= min_level:
                if entry.get("translation"):
                    parallel_data.append({
                        "mina": entry["text"],
                        "fr": entry["translation"],
                        "quality": quality,
                        "source": entry.get("source", "unknown"),
                    })

        with open(output_file, "w", encoding="utf-8") as f:
            for item in parallel_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Exporté {len(parallel_data)} paires parallèles vers {output_file}")
        return output_file

    def get_statistics(self) -> Dict:
        """Retourne les statistiques du corpus"""
        stats = {
            "total": len(self.pending) + len(self.validated) + len(self.rejected),
            "pending": len(self.pending),
            "validated": len(self.validated),
            "rejected": len(self.rejected),
            "validation_rate": 0.0,
        }

        total_annotated = len(self.validated) + len(self.rejected)
        if total_annotated > 0:
            stats["validation_rate"] = len(self.validated) / total_annotated

        # Qualité des validés
        quality_counts = {"good": 0, "medium": 0, "bad": 0}
        for entry in self.validated:
            q = entry.get("quality", "medium")
            quality_counts[q] = quality_counts.get(q, 0) + 1

        stats["quality_distribution"] = quality_counts

        return stats


class CLIAnnotator:
    """Interface CLI interactive pour annotation"""

    def __init__(self, manager: AnnotationManager = None):
        self.manager = manager or AnnotationManager()

    def print_entry(self, entry: AnnotationEntry):
        """Affiche une entrée"""
        print("\n" + "=" * 60)
        print(f"ID: {entry.id}")
        print(f"Langue: {entry.lang}")
        print(f"Source: {entry.source or 'N/A'}")
        print("-" * 60)
        print(f"TEXTE:\n{entry.text}")
        print("-" * 60)
        if entry.translation:
            print(f"TRADUCTION ACTUELLE:\n{entry.translation}")
        print("=" * 60)

    def run(self):
        """Lance l'interface d'annotation"""
        print("\n" + "=" * 60)
        print("ANNOTATEUR MINA-FRANCAIS")
        print("=" * 60)

        stats = self.manager.get_statistics()
        print(f"\nStatistiques:")
        print(f"  - En attente: {stats['pending']}")
        print(f"  - Validé: {stats['validated']}")
        print(f"  - Rejeté: {stats['rejected']}")

        while True:
            entry = self.manager.get_next()
            if not entry:
                print("\n[OK] Toutes les entrees ont ete annotees!")
                break

            self.print_entry(entry)

            print("\nOptions:")
            print("  [v] Valider (entrer traduction)")
            print("  [r] Rejeter")
            print("  [s] Passer (skip)")
            print("  [q] Quitter")

            choice = input("\n> ").strip().lower()

            if choice == "q":
                break
            elif choice == "v":
                translation = input("Traduction française: ").strip()
                quality = input("Qualité [good/medium/bad]: ").strip() or "medium"
                notes = input("Notes (optionnel): ").strip() or None

                self.manager.validate(
                    entry.id,
                    translation=translation,
                    quality=quality,
                    notes=notes,
                )
                print("[OK] Valide!")

            elif choice == "r":
                reason = input("Raison du rejet: ").strip()
                self.manager.reject(entry.id, reason)
                print("[X] Rejete!")

            elif choice == "s":
                self.manager.skip(entry.id)
                print("[>>] Passe")

        # Export final
        print("\nExport du corpus parallèle...")
        output = self.manager.export_parallel_corpus()
        print(f"Exporte vers: {output}")


def quick_add_entries():
    """Ajoute rapidement des entrées depuis un fichier"""
    manager = AnnotationManager()

    # Exemple avec phrases Mina de base
    base_phrases = [
        {"text": "Mɛlɔ", "lang": "mina", "translation": "Bonjour"},
        {"text": "Ntsɛ nyabi?", "lang": "mina", "translation": "Comment allez-vous?"},
        {"text": "Kofi wɔ nunana fefefia", "lang": "mina", "translation": "Kofi habite à Lomé"},
        {"text": "Nyabi mɔ", "lang": "mina", "translation": "Au revoir"},
        {"text": "Mi sɔ na?", "lang": "mina", "translation": "Comment vous appelez-vous?"},
        {"text": "Mɛ sɔ [nom]", "lang": "mina", "translation": "Je m'appelle [nom]"},
        {"text": "Nyametsɛ", "lang": "mina", "translation": "Merci"},
        {"text": "Ai" , "lang": "mina", "translation": "Oui"},
        {"text": "Ao" , "lang": "mina", "translation": "Non"},
        {"text": "Gbetato", "lang": "mina", "translation": "Boire"},
        {"text": "Dzakpɔ", "lang": "mina", "translation": "Manger"},
        {"text": "Azɔ", "lang": "mina", "translation": "Travailler"},
        {"text": "Nɔƒe", "lang": "mina", "translation": "Maison"},
        {"text": "Duke", "lang": "mina", "translation": "Eau"},
        {"text": "Fiafla", "lang": "mina", "translation": "Jour"},
    ]

    manager.add_batch(base_phrases)
    print(f"Ajouté {len(base_phrases)} entrées de base")

    return manager


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "add":
        # Mode ajout rapide
        quick_add_entries()
    elif len(sys.argv) > 1 and sys.argv[1] == "stats":
        # Mode statistiques
        manager = AnnotationManager()
        stats = manager.get_statistics()
        print(json.dumps(stats, indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "export":
        # Mode export
        manager = AnnotationManager()
        manager.export_parallel_corpus()
    else:
        # Mode interactif
        cli = CLIAnnotator()
        cli.run()