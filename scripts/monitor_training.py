"""
scripts/monitor_training.py - Tableau de bord pour surveiller l'entraînement Whisper
==================================================================================

Affiche en temps réel:
    - Progression (steps / total)
    - Loss actuelle
    - ETA (temps restant estimé)
    - Alertes critiques (NaN detection)

Usage:
    python scripts/monitor_training.py [--log-file logs/training.log]

Auteur: Claude Opus 4.8
Date: 2026-06-02
"""

import os
import sys
import re
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

# Fix UTF-8 pour Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_LOG_FILE = PROJECT_ROOT / "logs" / "training.log"
TRAINER_STATE_FILE = PROJECT_ROOT / "models" / "mina-whisper-v1" / "trainer_state.json"
REFRESH_INTERVAL = 2  # secondes

# États du terminal - Version Windows-compatible
class Colors:
    RESET = "" if os.name == 'nt' else "\033[0m"
    BOLD = "" if os.name == 'nt' else "\033[1m"
    DIM = "" if os.name == 'nt' else "\033[2m"
    RED = "" if os.name == 'nt' else "\033[91m"
    GREEN = "" if os.name == 'nt' else "\033[92m"
    YELLOW = "" if os.name == 'nt' else "\033[93m"
    BLUE = "" if os.name == 'nt' else "\033[94m"
    MAGENTA = "" if os.name == 'nt' else "\033[95m"
    CYAN = "" if os.name == 'nt' else "\033[96m"
    WHITE = "" if os.name == 'nt' else "\033[97m"
    BG_RED = "" if os.name == 'nt' else "\033[41m"
    BG_GREEN = "" if os.name == 'nt' else "\033[42m"

# Caractères de boîte compatibles Windows
BOX_TL = "+" if os.name == 'nt' else "+"
BOX_H = "-" if os.name == 'nt' else "-"
BOX_V = "|" if os.name == 'nt' else "|"
BOX_TR = "+" if os.name == 'nt' else "+"
BOX_BL = "+" if os.name == 'nt' else "+"
BOX_BR = "+" if os.name == 'nt' else "+"


class TrainingMonitor:
    """
    Moniteur temps réel pour l'entraînement Whisper
    """

    def __init__(self, log_file: Path = None):
        self.log_file = log_file or DEFAULT_LOG_FILE
        self.trainer_state_file = TRAINER_STATE_FILE

        # État de l'entraînement
        self.current_step = 0
        self.total_steps = 0
        self.current_loss = None
        self.best_loss = float('inf')
        self.learning_rate = 0
        self.epoch = 0
        self.start_time = None
        self.last_update = None

        # Historique des losses (pour graphique texte)
        self.loss_history = deque(maxlen=50)

        # Statut
        self.is_running = True
        self.has_nan_alert = False
        self.is_complete = False
        self.error_message = None

        # Fichier log handles
        self.log_file_pos = 0

    def clear_screen(self):
        """Efface l'écran du terminal"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_header(self):
        """Affiche l'en-tête du tableau de bord"""
        print(f"{Colors.CYAN}{Colors.BOLD}")
        print(f"{BOX_TL}{BOX_H * 78}{BOX_TR}")
        print(f"{BOX_V}{'':^78}{BOX_V}")
        print(f"{BOX_V}{'MINA-WHISPER TRAINING MONITOR - RTX 4060':^78}{BOX_V}")
        print(f"{BOX_V}{'':^78}{BOX_V}")
        print(f"{BOX_BL}{BOX_H * 78}{BOX_BR}")
        print(f"{Colors.RESET}")

    def print_status_bar(self):
        """Affiche la barre de progression"""

        # Calculer les statistiques
        if self.total_steps > 0:
            progress = min(self.current_step / self.total_steps, 1.0)
            progress_pct = progress * 100
            bar_len = 50
            filled = int(bar_len * progress)
            if os.name == 'nt':
                bar = "#" * filled + "." * (bar_len - filled)
            else:
                bar = "█" * filled + "░" * (bar_len - filled)
        else:
            progress_pct = 0
            if os.name == 'nt':
                bar = "." * 50
            else:
                bar = "░" * 50
            bar_len = 50
            filled = 0

        # ETA calculé
        eta = self._calculate_eta()

        print(f"{Colors.BLUE}{'─' * 78}{Colors.RESET}")
        print(f"  {Colors.BOLD}Progression:{Colors.RESET} [{bar}] {progress_pct:.1f}%")
        print(f"  {Colors.BOLD}Steps:{Colors.RESET} {self.current_step:,} / {self.total_steps:,}")
        print(f"  {Colors.BOLD}Epoch:{Colors.RESET} {self.epoch}")
        print(f"  {Colors.BOLD}ETA:{Colors.RESET} {eta}")
        print(f"{Colors.BLUE}{'─' * 78}{Colors.RESET}")

    def print_metrics(self):
        """Affiche les métriques principales"""
        print()
        print(f"  {Colors.CYAN}{Colors.BOLD}+{'-' * 41}+{Colors.RESET}")
        print(f"  {Colors.CYAN}{Colors.BOLD}|          METRIQUES ACTUELLES          |{Colors.RESET}")
        print(f"  {Colors.CYAN}{Colors.BOLD}+{'-' * 41}+{Colors.RESET}")
        print()

        # Loss
        if self.current_loss is not None:
            if self.has_nan_alert:
                loss_color = Colors.RED
                loss_icon = "⚠️ NAN!"
            elif self.current_loss < self.best_loss:
                loss_color = Colors.GREEN
                loss_icon = "↓"
            else:
                loss_color = Colors.YELLOW
                loss_icon = "→"
            print(f"  {Colors.BOLD}Loss:{Colors.RESET} {loss_color}{loss_icon} {self.current_loss:.6f}{Colors.RESET}")
        else:
            print(f"  {Colors.BOLD}Loss:{Colors.RESET} {Colors.DIM}En attente...{Colors.RESET}")

        # Best Loss
        if self.best_loss < float('inf'):
            print(f"  {Colors.BOLD}Best Loss:{Colors.RESET} {Colors.GREEN}✓ {self.best_loss:.6f}{Colors.RESET}")
        else:
            print(f"  {Colors.BOLD}Best Loss:{Colors.RESET} {Colors.DIM}-{Colors.RESET}")

        # Learning Rate
        print(f"  {Colors.BOLD}Learning Rate:{Colors.RESET} {self.learning_rate:.2e}")

        print()

    def print_loss_graph(self):
        """Affiche un graphique textuel de la loss"""
        if len(self.loss_history) < 2:
            return

        print(f"  {Colors.BOLD}Historique Loss (50 derniers):{Colors.RESET}")

        # Normaliser les valeurs pour l'affichage
        min_loss = min(self.loss_history)
        max_loss = max(self.loss_history)
        range_loss = max_loss - min_loss if max_loss != min_loss else 1

        graph_lines = []
        for i in range(6, -1, -1):
            threshold = min_loss + (range_loss * i / 6)
            if os.name == 'nt':
                line = f"  {threshold:8.4f} |"
            else:
                line = f"  {threshold:8.4f} │"

            for loss_val in self.loss_history:
                if loss_val >= threshold:
                    if os.name == 'nt':
                line += "#"
            else:
                line += "█"
                else:
                    line += " "

            # Ajouter l'échelle
            if i == 6:
                line += f" {Colors.GREEN}max{Colors.RESET}"
            elif i == 0:
                line += f" {Colors.RED}min{Colors.RESET}"

            graph_lines.append(line)

        for line in graph_lines:
            print(line)

        # Baseline
        baseline = " " * 16 + "+" + "-" * min(len(self.loss_history), 50) + "+"
        print(f"  {Colors.DIM}{' ' * 16}{baseline}{Colors.RESET}")
        print()

    def print_alerts(self):
        """Affiche les alertes actives"""
        if self.has_nan_alert:
            print(f"{Colors.BG_RED}{Colors.WHITE}{Colors.BOLD}")
            print(f"+{'-' * 78}+")
            print(f"|  ALERTE CRITIQUE: NaN detecte dans la Loss!                             |")
            print(f"|  L'entrainement peut etre compromis                                    |")
            print(f"|  Solutions: reduire LR, verifier les donnees, activer gradient clipping |")
            print(f"+{'-' * 78}+")
            print(f"{Colors.RESET}")

        if self.error_message:
            print(f"{Colors.RED}{Colors.BOLD}ERREUR: {self.error_message}{Colors.RESET}")
            print()

    def print_completion_summary(self):
        """Affiche le résumé de fin d'entraînement"""
        if not self.is_complete:
            return

        elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)

        print()
        print(f"{Colors.GREEN}{Colors.BOLD}")
        print(f"+{'-' * 78}+")
        print(f"|                    ENTRAINEMENT TERMINE AVEC SUCCES                      |")
        print(f"+{'-' * 78}+")
        print(f"|  Duree totale: {elapsed}                                           |")
        print(f"|  Steps completes: {self.current_step:,}                                          |")
        print(f"|  Best Loss: {self.best_loss:.6f}                                                   |")
        print(f"|                                                                          |")
        print(f"|  Modele sauvegarde dans: models/mina-whisper-v1/                          |")
        print(f"+{'-' * 78}+")
        print(f"{Colors.RESET}")

    def print_footer(self):
        """Affiche le pied de page"""
        print()
        print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")
        print(f"  Log file: {self.log_file}")
        print(f"  Trainer's state: {self.trainer_state_file}")
        print(f"  Last update: {datetime.now().strftime('%H:%M:%S')}")
        print(f"  Press Ctrl+C to stop monitoring")
        print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")

    def _calculate_eta(self) -> str:
        """Calcule le temps restant estimé"""
        if self.current_step == 0 or self.start_time is None:
            return "Calcul en cours..."

        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed < 1:
            return "Calcul en cours..."

        steps_per_second = self.current_step / elapsed
        if steps_per_second < 0.001:
            return "Très lent..."

        remaining_steps = self.total_steps - self.current_step
        if remaining_steps <= 0:
            return "Terminé!"

        remaining_seconds = remaining_steps / steps_per_second
        eta_delta = timedelta(seconds=int(remaining_seconds))

        # Formatage lisible
        if eta_delta.total_seconds() < 60:
            return f"{int(eta_delta.total_seconds())}s"
        elif eta_delta.total_seconds() < 3600:
            minutes = int(eta_delta.total_seconds() / 60)
            return f"{minutes}min {int(eta_delta.total_seconds() % 60)}s"
        else:
            hours = int(eta_delta.total_seconds() / 3600)
            minutes = int((eta_delta.total_seconds() % 3600) / 60)
            return f"{hours}h {minutes}min"

    def _parse_log_file(self):
        """Parse le fichier de log pour extraire les métriques"""
        if not self.log_file.exists():
            return

        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                # Lire seulement les nouvelles lignes
                f.seek(self.log_file_pos)
                new_lines = f.readlines()
                self.log_file_pos = f.tell()

                for line in new_lines:
                    self._parse_log_line(line)
        except Exception as e:
            pass  # Ignore les erreurs de lecture

    def _parse_log_line(self, line: str):
        """Parse une ligne de log"""

        # Détecter le démarrage (capturer le temps)
        if 'Starting training' in line or 'Start training' in line or 'DÉMARRAGE' in line.upper():
            if self.start_time is None:
                self.start_time = datetime.now()

        # Extraire les steps
        step_match = re.search(r'step\s+(\d+)[/\\](\d+)', line, re.IGNORECASE)
        if step_match:
            self.current_step = int(step_match.group(1))
            self.total_steps = int(step_match.group(2))

        # Extraire la loss (formats multiples)
        loss_patterns = [
            r'loss[":\s]+([0-9.]+)',
            r'"loss":\s*([0-9.]+)',
            r'loss\s*[:=]\s*([0-9.]+)',
            r',\s*loss[":\s]+([0-9.]+)',
        ]

        for pattern in loss_patterns:
            loss_match = re.search(pattern, line, re.IGNORECASE)
            if loss_match:
                loss_val = float(loss_match.group(1))

                # Détecter NaN
                if loss_val != loss_val:  # NaN check
                    self.has_nan_alert = True
                    self.current_loss = float('nan')
                else:
                    self.current_loss = loss_val
                    self.loss_history.append(loss_val)
                    if loss_val < self.best_loss:
                        self.best_loss = loss_val
                break

        # Extraire le learning rate
        lr_match = re.search(r'lr[":\s]+([0-9.e-]+)', line, re.IGNORECASE)
        if lr_match:
            self.learning_rate = float(lr_match.group(1))

        # Extraire l'epoch
        epoch_match = re.search(r'epoch[":\s]+([0-9.]+)', line, re.IGNORECASE)
        if epoch_match:
            self.epoch = float(epoch_match.group(1))

        # Détecter les erreurs
        if 'error' in line.lower() or 'exception' in line.lower() or 'failed' in line.lower():
            if 'nan' in line.lower():
                self.has_nan_alert = True

    def _parse_trainer_state(self):
        """Parse le fichier trainer_state.json"""
        if not self.trainer_state_file.exists():
            return

        try:
            with open(self.trainer_state_file, 'r') as f:
                state = json.load(f)

            # Extraire les métriques
            if 'log_history' in state and state['log_history']:
                last_log = state['log_history'][-1]

                if 'step' in last_log:
                    self.current_step = last_log['step']
                if 'loss' in last_log:
                    loss_val = last_log['loss']
                    if loss_val != loss_val:  # NaN
                        self.has_nan_alert = True
                        self.current_loss = float('nan')
                    else:
                        self.current_loss = loss_val
                        self.loss_history.append(loss_val)
                        if loss_val < self.best_loss:
                            self.best_loss = loss_val

                if 'learning_rate' in last_log:
                    self.learning_rate = last_log['learning_rate']

                if 'epoch' in last_log:
                    self.epoch = last_log['epoch']

            # Vérifier le total_steps
            if 'max_steps' in state:
                self.total_steps = state['max_steps']

            # Vérifier si terminé
            if 'is_local_process_zero' in state and state.get('epoch', 0) >= 1:
                if self.current_step >= self.total_steps:
                    self.is_complete = True

        except Exception as e:
            pass  # Ignore les erreurs

    def check_training_status(self):
        """Vérifie si l'entraînement est terminé"""
        # Vérifier via trainer_state.json
        self._parse_trainer_state()

        # Vérifier via les logs
        self._parse_log_file()

        # Si trainer_state existe et epoch >= 3, on considère comme terminé
        if self.trainer_state_file.exists():
            try:
                with open(self.trainer_state_file, 'r') as f:
                    state = json.load(f)
                if state.get('epoch', 0) >= 3:
                    self.is_complete = True
            except:
                pass

        # Vérifier via les processus
        if not self._is_training_running():
            self.is_complete = True

    def _is_training_running(self) -> bool:
        """Vérifie si le processus d'entraînement est encore actif"""
        if os.name == 'nt':
            # Windows
            import subprocess
            result = subprocess.run(
                ['tasklist'], capture_output=True, text=True
            )
            return 'python' in result.stdout.lower()
        else:
            # Unix
            import subprocess
            result = subprocess.run(
                ['ps', 'aux'], capture_output=True, text=True
            )
            return 'train_whisper' in result.stdout

    def run(self):
        """Boucle principale du moniteur"""
        self.clear_screen()
        self.print_header()

        while self.is_running:
            # Vérifier le statut de l'entraînement
            self.check_training_status()

            # Mettre à jour l'affichage
            self.clear_screen()
            self.print_header()
            self.print_status_bar()
            self.print_metrics()
            self.print_loss_graph()
            self.print_alerts()

            if self.is_complete:
                self.print_completion_summary()
                print()
                print("Appuyez sur Ctrl+C pour quitter ou attendez 10 secondes...")
                time.sleep(10)
                break

            self.print_footer()

            # Attendre avant la prochaine mise à jour
            for _ in range(REFRESH_INTERVAL * 5):
                time.sleep(0.2)
                # Vérifier régulièrement si l'entraînement est terminé
                if not self._is_training_running():
                    self.is_complete = True
                    break


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Monitoring pour l'entraînement Whisper")
    parser.add_argument(
        '--log-file',
        type=str,
        default=str(DEFAULT_LOG_FILE),
        help='Chemin vers le fichier de log'
    )

    args = parser.parse_args()

    monitor = TrainingMonitor(log_file=Path(args.log_file))

    try:
        monitor.run()
    except KeyboardInterrupt:
        print()
        print(f"{Colors.YELLOW}Monitoring interrompu par l'utilisateur{Colors.RESET}")
        print(f"{Colors.CYAN}L'entraînement continue en arrière-plan{Colors.RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()