"""
scripts/index_dataset.py - Indexation et unification du dataset Common Voice Mina
===================================================================================

Ce script fusionne les fichiers TSV du dataset Common Voice (gej) en un corpus JSONL
unifié pour l'entraînement de modèles STT/TTS.

Structure attendue:
    data/cv-corpus-25.0-2026-03-09/
        clips/              <- fichiers audio MP3
        train.tsv           <- données d'entraînement
        dev.tsv             <- données de validation
        test.tsv            <- données de test
        validated.tsv       <- données validées (optionnel)

Output:
    data/corpus/mina_full_dataset.jsonl

Usage:
    python scripts/index_dataset.py

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configuration du logging
from loguru import logger

# =============================================================================
# CONFIGURATION
# =============================================================================

# Racine du projet
PROJECT_ROOT = Path(__file__).parent.parent
CV_DATASET_DIR = PROJECT_ROOT / "data" / "cv-corpus-25.0-2026-03-09"
# Les fichiers TSV sont dans le sous-dossier 'gej/'
GEJ_DIR = CV_DATASET_DIR / "gej"
CLIPS_DIR = GEJ_DIR / "clips"
OUTPUT_FILE = PROJECT_ROOT / "data" / "corpus" / "mina_full_dataset.jsonl"

# Fichiers TSV à traiter (dans le dossier gej/)
TSV_SPLITS = {
    "train": "train.tsv",
    "dev": "dev.tsv",
    "test": "test.tsv",
}

# Optionnel: inclure validated.tsv
INCLUDE_VALIDATED = True
VALIDATED_FILE = "validated.tsv"

# Colonnes attendues dans les fichiers TSV (Common Voice 25.0)
EXPECTED_COLUMNS = ["client_id", "path", "sentence", "up_votes", "down_votes", "age",
                   "gender", "accents", "variant", "locale", "segment"]

# Encodage à tester (par ordre de priorité)
ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]


# =============================================================================
# CLASSE D'INDEXATION
# =============================================================================

class CommonVoiceIndexer:
    """
    Indexeur robuste pour le dataset Common Voice Mina (gej)
    """

    def __init__(self, dataset_dir: Path, gej_dir: Path, clips_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.gej_dir = Path(gej_dir)  # Dossier gej où se trouvent les TSV
        self.clips_dir = Path(clips_dir)
        self.output_file = None

        # Statistiques
        self.stats = defaultdict(lambda: {
            "lines_read": 0,
            "valid_entries": 0,
            "missing_audio": 0,
            "empty_text": 0,
            "invalid_encoding": 0,
        })

        # Entrées valides
        self.entries = []

        # Fichiers audio manquants (pour rapport)
        self.missing_audios = []

    def detect_encoding(self, filepath: Path) -> str:
        """
        Détecte automatiquement l'encodage du fichier

        Args:
            filepath: Chemin vers le fichier TSV

        Returns:
            Encodage détecté
        """
        for encoding in ENCODINGS_TO_TRY:
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    # Lire les 3 premières lignes pour tester
                    for _ in range(3):
                        f.readline()
                logger.debug(f"Encodage détecté pour {filepath.name}: {encoding}")
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue

        logger.warning(f"Impossible de détecter l'encodage, utilisation de utf-8")
        return "utf-8"

    def parse_tsv(self, filepath: Path, split: str) -> list:
        """
        Parse un fichier TSV Common Voice

        Args:
            filepath: Chemin vers le fichier TSV
            split: Nom du split (train/dev/test)

        Returns:
            Liste de dictionnaires avec les entrées
        """
        entries = []
        encoding = self.detect_encoding(filepath)

        try:
            with open(filepath, "r", encoding=encoding, errors="replace") as f:
                # Lire l'en-tête
                header_line = f.readline().strip()
                headers = self._parse_tsv_line(header_line)

                # Log des colonnes disponibles
                logger.info(f"Colonnes détectées: {headers}")

                # Index des colonnes utiles
                col_map = self._get_column_indices(headers)

                for line_num, line in enumerate(f, start=2):
                    values = self._parse_tsv_line(line.rstrip('\n\r'))

                    if len(values) < len(headers):
                        logger.debug(f"Ligne {line_num} tronquée, ignorée")
                        continue

                    entry = self._extract_entry(values, headers, col_map, split, line_num)
                    if entry:
                        entries.append(entry)

        except FileNotFoundError:
            logger.error(f"Fichier non trouvé: {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors du parsing de {filepath}: {e}")

        return entries

    def _parse_tsv_line(self, line: str) -> list:
        """Parse une ligne TSV en gérant les champs vides"""
        if not line:
            return []
        return line.split("\t")

    def _get_column_indices(self, headers: list) -> dict:
        """
        Retourne les indices des colonnes utiles

        Common Voice 25.0 utilise 'path' et 'sentence'
        """
        col_map = {}

        # Map des noms de colonnes alternatifs
        column_names = {
            "path": ["path", "audio_path", "file"],
            "sentence": ["sentence", "text", "transcription", "description"],
            "client_id": ["client_id", "clientId"],
            "up_votes": ["up_votes", "upvotes"],
            "down_votes": ["down_votes", "downvotes"],
        }

        for target, alternatives in column_names.items():
            for alt in alternatives:
                if alt in headers:
                    col_map[target] = headers.index(alt)
                    break

        return col_map

    def _extract_entry(self, values: list, headers: list, col_map: dict,
                      split: str, line_num: int) -> dict:
        """
        Extrait et valide une entrée depuis une ligne TSV
        """
        stats = self.stats[split]
        stats["lines_read"] += 1

        # Extraire les champs
        audio_path = self._safe_get(values, col_map.get("path"))
        text = self._safe_get(values, col_map.get("sentence"))

        # Vérifier que le texte n'est pas vide
        if not text or not text.strip():
            stats["empty_text"] += 1
            return None

        # Vérifier que le chemin audio est présent
        if not audio_path:
            stats["missing_audio"] += 1
            return None

        # Construire le chemin relatif
        relative_path = f"gej/clips/{audio_path}" if not audio_path.startswith("gej/clips/") else audio_path

        # Vérifier que le fichier audio existe
        full_audio_path = self.dataset_dir / relative_path
        if not full_audio_path.exists():
            stats["missing_audio"] += 1
            if len(self.missing_audios) < 100:  # Limiter la liste
                self.missing_audios.append(str(relative_path))
            return None

        stats["valid_entries"] += 1

        return {
            "text": text.strip(),
            "audio_path": relative_path,
            "split": split,
            "validated": True,
        }

    def _safe_get(self, values: list, index: int) -> str:
        """Récupère une valeur en toute sécurité"""
        if index is None or index >= len(values):
            return ""
        return values[index].strip() if values[index] else ""

    def index_split(self, split: str, filename: str) -> int:
        """
        Indexe un fichier TSV et retourne le nombre d'entrées

        Args:
            split: Nom du split (train/dev/test)
            filename: Nom du fichier TSV

        Returns:
            Nombre d'entrées ajoutées
        """
        # Les fichiers TSV sont dans le dossier gej/
        filepath = self.gej_dir / filename

        if not filepath.exists():
            logger.warning(f"Fichier TSV non trouvé: {filepath}")
            return 0

        logger.info(f"Indexation de {filename}...")

        entries = self.parse_tsv(filepath, split)
        self.entries.extend(entries)

        return len(entries)

    def run(self, output_file: Path) -> dict:
        """
        Exécute l'indexation complète

        Args:
            output_file: Chemin du fichier JSONL de sortie

        Returns:
            Statistiques complètes
        """
        logger.info("=" * 60)
        logger.info("INDEXATION DU DATASET COMMON VOICE MINA (gej)")
        logger.info("=" * 60)
        logger.info(f"Dataset: {self.dataset_dir}")
        logger.info(f"Output: {output_file}")
        logger.info("")

        # Créer le répertoire de sortie
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Indexer chaque split
        total_entries = 0
        for split, filename in TSV_SPLITS.items():
            count = self.index_split(split, filename)
            total_entries += count

        # Indexer validated.tsv si demande
        if INCLUDE_VALIDATED:
            # Les fichiers TSV sont dans le dossier gej/
            validated_path = self.gej_dir / VALIDATED_FILE
            if validated_path.exists():
                count = self.index_split("validated", VALIDATED_FILE)
                total_entries += count

        # Écrire le fichier JSONL
        logger.info("")
        logger.info("Écriture du fichier JSONL...")

        with open(output_file, "w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(f"✅ {len(self.entries)} entrees ecrites")

        # Calculer les statistiques finales
        return self._generate_report(output_file)

    def _generate_report(self, output_file: Path) -> dict:
        """Génère le rapport final"""
        total_lines = sum(s["lines_read"] for s in self.stats.values())
        total_valid = sum(s["valid_entries"] for s in self.stats.values())
        total_missing = sum(s["missing_audio"] for s in self.stats.values())
        total_empty = sum(s["empty_text"] for s in self.stats.values())

        # Taille du fichier
        file_size = output_file.stat().st_size if output_file.exists() else 0
        file_size_mb = file_size / (1024 * 1024)

        # Taille estimée des fichiers audio
        audio_size = self._estimate_audio_size()

        report = {
            "dataset": "common-voice-mina-gej",
            "version": "25.0-2026-03-09",
            "generated_at": datetime.now().isoformat(),
            "splits": dict(self.stats),
            "totals": {
                "lines_processed": total_lines,
                "valid_entries": total_valid,
                "missing_audio": total_missing,
                "empty_text": total_empty,
            },
            "output_file": str(output_file),
            "output_size_mb": round(file_size_mb, 2),
            "estimated_audio_size_gb": round(audio_size / (1024 * 1024 * 1024), 2),
        }

        # Afficher le rapport
        self._print_report(report)

        return report

    def _estimate_audio_size(self) -> int:
        """Estime la taille totale des fichiers audio"""
        total_size = 0
        for entry in self.entries:
            audio_path = self.dataset_dir / entry["audio_path"]
            if audio_path.exists():
                total_size += audio_path.stat().st_size
        return total_size

    def _print_report(self, report: dict):
        """Affiche le rapport formaté"""
        print()
        print("=" * 60)
        print("RAPPORT D'INDEXATION")
        print("=" * 60)
        print()

        print(f"Dataset: common-voice-mina-gej")
        print(f"Version: {report['version']}")
        print(f"Genere: {report['generated_at']}")
        print()

        print("+---------+------------+------------+------------+------------+------------+")
        print("| Split   | Lignes     | Valides    | Manquants  | Texte vide | Validation |")
        print("+---------+------------+------------+------------+------------+------------+")

        for split in ["train", "dev", "test", "validated"]:
            if split in report["splits"]:
                s = report["splits"][split]
                pct = (s["valid_entries"] / s["lines_read"] * 100) if s["lines_read"] > 0 else 0
                print(f"| {split:7} | {s['lines_read']:10} | {s['valid_entries']:10} | {s['missing_audio']:10} | {s['empty_text']:10} | {pct:5.1f}%  |")

        print("+---------+------------+------------+------------+------------+------------+")
        print()

        totals = report["totals"]
        print(f"Total des lignes traitees: {totals['lines_processed']:,}")
        print(f"Entrees valides: {totals['valid_entries']:,}")
        print(f"Fichiers audio manquants: {totals['missing_audio']:,}")
        print(f"Entrees avec texte vide: {totals['empty_text']:,}")
        print()

        print(f"Fichier JSONL genere: {report['output_file']}")
        print(f"Taille JSONL: {report['output_size_mb']} MB")
        print(f"Taille estimee audio: {report['estimated_audio_size_gb']} Go")
        print()

        if totals["missing_audio"] > 0:
            print(f"Attention: {totals['missing_audio']} fichiers audio non trouves")
            if self.missing_audios:
                print(f"   Exemples: {self.missing_audios[:5]}")

        print()
        print("=" * 60)
        print("Indexation terminee avec succes!")
        print("=" * 60)


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    """Point d'entrée principal"""

    logger.info("Demarrage de l'indexation...")

    # Vérifier que le dataset existe
    if not CV_DATASET_DIR.exists():
        logger.error(f"Dataset non trouvé: {CV_DATASET_DIR}")
        print(f"""
ERREUR: Dataset non trouvé

Le dossier {CV_DATASET_DIR} n'existe pas.

Veuillez verifier que:
1. Le dataset Common Voice Mina (gej) est bien telecharge
2. Le dossier s'appelle 'cv-corpus-25.0-2026-03-09'
3. Il contient les fichiers 'train.tsv', 'dev.tsv', 'test.tsv' dans le sous-dossier 'gej/'

Structure attendue:
    data/
        cv-corpus-25.0-2026-03-09/
            gej/
                clips/
                    fichier1.mp3
                    fichier2.mp3
                    ...
                train.tsv
                dev.tsv
                test.tsv
                validated.tsv
""")
        sys.exit(1)

    # Vérifier le contenu du dataset
    if not GEJ_DIR.exists():
        logger.error(f"Repertoire gej non trouve: {GEJ_DIR}")
        sys.exit(1)

    clips_dir = GEJ_DIR / "clips"
    if not clips_dir.exists():
        logger.error(f"Repertoire clips non trouve: {clips_dir}")
        sys.exit(1)

    # Compter les fichiers audio
    audio_files = list(clips_dir.glob("*.mp3")) + list(clips_dir.glob("*.mp4")) + list(clips_dir.glob("*.ogg"))
    logger.info(f"Fichiers audio trouves: {len(audio_files)}")

    # Creer l'indexeur
    indexer = CommonVoiceIndexer(
        dataset_dir=CV_DATASET_DIR,
        gej_dir=GEJ_DIR,
        clips_dir=clips_dir
    )

    # Executer l'indexation
    report = indexer.run(OUTPUT_FILE)

    # Sauvegarder le rapport
    report_file = PROJECT_ROOT / "data" / "corpus" / "dataset_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Rapport sauvegarde: {report_file}")

    return report


if __name__ == "__main__":
    main()