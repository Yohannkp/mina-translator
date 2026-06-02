#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outil d'annotation pour valider les traductions Mina-Français.
Interface terminal simple avec navigation au clavier.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import sys


@dataclass
class Annotation:
    """Structure de données pour une annotation de traduction."""
    fr: str
    mina: str
    mina_corrected: Optional[str] = None
    validated: bool = False
    validator_notes: Optional[str] = None


class MinaAnnotator:
    """Outil d'annotation pour les traductions Mina-Français."""

    # Caractères Mina spéciaux avec leurs combinaisons
    MINA_TONES = {
        'a': 'a', 'á': 'a', 'à': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a',
        'e': 'e', 'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'ɛ': 'ɛ', 'ɛ́': 'ɛ', 'ɛ̀': 'ɛ', 'ɛ̂': 'ɛ', 'ɛ̃': 'ɛ', 'ɛ̈': 'ɛ',
        'i': 'i', 'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
        'o': 'o', 'ó': 'o', 'ò': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o',
        'ɔ': 'ɔ', 'ɔ́': 'ɔ', 'ɔ̀': 'ɔ', 'ɔ̂': 'ɔ', 'ɔ̃': 'ɔ', 'ɔ̈': 'ɔ',
        'u': 'u', 'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
        'ɖ': 'ɖ', 'ɖ́': 'ɖ', 'ɖ̀': 'ɖ', 'ɖ̂': 'ɖ', 'ɖ̃': 'ɖ', 'ɖ̈': 'ɖ',
        'n': 'n', 'ń': 'n', 'ǹ': 'n', 'n̂': 'n', 'ñ': 'n', 'n̈': 'n',
        'm': 'm', 'ḿ': 'm', 'm̀': 'm', 'm̂': 'm', 'm̃': 'm', 'm̈': 'm',
    }

    def __init__(self, input_file: str, output_file: Optional[str] = None):
        """
        Initialise l'outil d'annotation.

        Args:
            input_file: Chemin vers le fichier JSONL d'entrée
            output_file: Chemin vers le fichier JSONL de sortie (par défaut: input_file.annotated.jsonl)
        """
        self.input_file = Path(input_file)
        self.output_file = Path(output_file) if output_file else self.input_file.with_suffix('.annotated.jsonl')

        self.annotations: list[Annotation] = []
        self.current_index = 0
        self.stats = {
            'validated': 0,
            'corrected': 0,
            'skipped': 0
        }

        self._load_data()

    def _load_data(self) -> None:
        """Charge les données depuis le fichier d'entrée."""
        if not self.input_file.exists():
            print(f"Erreur: Le fichier '{self.input_file}' n'existe pas.")
            sys.exit(1)

        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        # Support both direct data and wrapped format
                        if 'fr' in data and 'mina' in data:
                            self.annotations.append(Annotation(
                                fr=data['fr'],
                                mina=data['mina']
                            ))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Ligne invalide ignorée: {e}")

        if not self.annotations:
            print("Erreur: Aucune donnée valide trouvée dans le fichier.")
            sys.exit(1)

        print(f"Chargé {len(self.annotations)} traductions depuis '{self.input_file}'")

    def _clear_screen(self) -> None:
        """Efface l'écran du terminal."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _get_existing_annotation(self) -> Optional[Annotation]:
        """Récupère une annotation existante si le fichier de sortie existe déjà."""
        if not self.output_file.exists():
            return None

        with open(self.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        # Chercher une correspondance par la phrase française
                        if data.get('fr') == self.annotations[self.current_index].fr:
                            return Annotation(
                                fr=data['fr'],
                                mina=data['mina'],
                                mina_corrected=data.get('mina_corrected'),
                                validated=data.get('validated', False),
                                validator_notes=data.get('validator_notes')
                            )
                    except json.JSONDecodeError:
                        continue
        return None

    def _display_header(self) -> None:
        """Affiche l'en-tête de l'interface."""
        total = len(self.annotations)
        current = self.current_index + 1
        validated = self.stats['validated']
        corrected = self.stats['corrected']
        skipped = self.stats['skipped']

        print("=" * 60)
        print("  OUTIL D'ANNOTATION MINA-FRANÇAIS")
        print("=" * 60)
        print(f"  Progression: {current}/{total} | "
              f"Validées: {validated} | "
              f"Corrigées: {corrected} | "
              f"Passées: {skipped}")
        print("=" * 60)

    def _display_annotation(self, annotation: Annotation) -> None:
        """Affiche une annotation à réviser."""
        print()
        print("-" * 60)
        print("  📝 FRANÇAIS:")
        print(f"  {annotation.fr}")
        print()
        print("  🌐 MINA:")
        print(f"  {annotation.mina}")
        print("-" * 60)

    def _display_help(self) -> None:
        """Affiche l'aide des touches disponibles."""
        print()
        print("  Touches:")
        print("    [o] OK           - Valider la traduction")
        print("    [c] Corriger     - Proposer une correction")
        print("    [s] Skip         - Passer à la suivante")
        print("    [q] Quitter      - Sauvegarder et quitter")
        print("    [r] Revenir      - Recommencer depuis le début")
        print()

    def _handle_ok(self) -> None:
        """Valide la traduction actuelle."""
        annotation = self.annotations[self.current_index]
        annotation.validated = True
        self.stats['validated'] += 1
        self._save_annotation(annotation)
        print("  ✓ Traduction validée!")

    def _handle_correct(self) -> None:
        """Propose une correction pour la traduction actuelle."""
        annotation = self.annotations[self.current_index]
        print()
        print("  Proposition de correction (Enter pour garder '{0}'):".format(annotation.mina))
        correction = input("  > ").strip()

        if correction:
            annotation.mina_corrected = correction
            print("  Notes (optionnel, Enter pour passer):")
            notes = input("  > ").strip()
            if notes:
                annotation.validator_notes = notes
        else:
            annotation.mina_corrected = annotation.mina

        annotation.validated = True
        self.stats['validated'] += 1
        self.stats['corrected'] += 1
        self._save_annotation(annotation)
        print("  ✓ Correction enregistrée!")

    def _handle_skip(self) -> None:
        """Passe à la traduction suivante sans annoter."""
        self.stats['skipped'] += 1
        print("  → Traduction passée")

    def _save_annotation(self, annotation: Annotation) -> None:
        """Sauvegarde une annotation dans le fichier de sortie."""
        data = asdict(annotation)
        # Remove None values for cleaner output
        data = {k: v for k, v in data.items() if v is not None}

        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def _handle_quit(self) -> None:
        """Sauvegarde et quitte le programme."""
        print()
        print("=" * 60)
        print("  RÉSUMÉ DE LA SESSION")
        print("=" * 60)
        print(f"  Total traitées: {self.current_index + 1}/{len(self.annotations)}")
        print(f"  Validées:       {self.stats['validated']}")
        print(f"  Corrigées:      {self.stats['corrected']}")
        print(f"  Passées:        {self.stats['skipped']}")
        print()
        print(f"  Annotations sauvegardées dans: {self.output_file}")
        print("=" * 60)
        print("  Au revoir!")
        sys.exit(0)

    def run(self) -> None:
        """Lance l'interface d'annotation."""
        self._clear_screen()

        print("Bienvenue dans l'outil d'annotation Mina-Français!")
        print(f"Fichier d'entrée: {self.input_file}")
        print(f"Fichier de sortie: {self.output_file}")
        print()

        input("Appuyez sur Entrée pour commencer...")

        while True:
            self._clear_screen()

            if self.current_index >= len(self.annotations):
                print("Toutes les traductions ont été traitées!")
                self._handle_quit()

            annotation = self.annotations[self.current_index]

            self._display_header()
            self._display_annotation(annotation)
            self._display_help()

            # Get user input
            try:
                user_input = input("  Votre choix: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print()
                self._handle_quit()

            if user_input == 'o':
                self._handle_ok()
            elif user_input == 'c':
                self._handle_correct()
            elif user_input == 's':
                self._handle_skip()
            elif user_input == 'q':
                self._handle_quit()
            elif user_input == 'r':
                self.current_index = 0
                self.stats = {'validated': 0, 'corrected': 0, 'skipped': 0}
                if self.output_file.exists():
                    self.output_file.unlink()
                print("  → Redémarrage depuis le début")
                input("  Appuyez sur Entrée...")
                continue
            else:
                print("  ⚠ Commande non reconnue. Utilisez: o, c, s, q ou r")
                input("  Appuyez sur Entrée pour continuer...")

            self.current_index += 1


def create_sample_file(filename: str = "sample_mina.jsonl") -> None:
    """Crée un fichier d'exemple pour tester l'outil."""
    samples = [
        {"fr": "Bonjour, comment allez-vous?", "mina": "Wǔtɛ̀n, àdogɔ a fà?"},
        {"fr": "Je suis content de vous voir.", "mina": "Mǐwě bǐ nǔwǔ dó."},
        {"fr": "Où est le marché?", "mina": "Tɔkʊ́ nǔ bǐ yɛ̀?"},
        {"fr": "Le soleil se lève à l'est.", "mina": "Yǒkʊ́ sùnù tɛ́ wǔjǔ."},
        {"fr": "J'ai faim et soif.", "mina": "Mǐnǔ kpɔ kpɔ tǝ̀ lǝ́."},
        {"fr": "Merci beaucoup pour votre aide.", "mina": "Ɖǒwǔ kɛ́ yí mǔwǔ."},
        {"fr": "La pluie tombe ce soir.", "mina": "Jìmǔ wǔlǝ́ ɖǐ."},
        {"fr": "Il fait chaud aujourd'hui.", "mina": "Yǒkʊ́ bǐ tǝ́ wǔdǒ."},
        {"fr": "Je ne comprends pas.", "mina": "Mǐɖà kùtù mǔwǔ."},
        {"fr": "Parlez-vous mina?", "mina": "Ɓǒ nǔ ɖǐ Mina tǝ̀?"},
    ]

    with open(filename, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Fichier d'exemple créé: {filename}")


def main():
    """Point d'entrée principal du programme."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Outil d'annotation pour valider les traductions Mina-Français"
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        help="Fichier JSONL contenant les traductions à annoter"
    )
    parser.add_argument(
        '-o', '--output',
        help="Fichier de sortie pour les annotations"
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help="Créer un fichier d'exemple pour tester"
    )
    parser.add_argument(
        '--sample-file',
        default="sample_mina.jsonl",
        help="Nom du fichier d'exemple à créer"
    )

    args = parser.parse_args()

    if args.sample:
        create_sample_file(args.sample_file)
        return

    if not args.input_file:
        parser.print_help()
        print("\nExemples:")
        print("  python mina_annotator.py traductions.jsonl")
        print("  python mina_annotator.py --sample  # Créer un fichier test")
        print("  python mina_annotator.py -o output.jsonl input.jsonl")
        return

    annotator = MinaAnnotator(args.input_file, args.output)
    annotator.run()


if __name__ == "__main__":
    main()