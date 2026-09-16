"""
crowdsource_app.py - Application Streamlit pour le crowdsourcing Mina → Français
================================================================================

Une application où les utilisateurs :
1. Écoutent une phrase en Mina (audio aléatoire)
2. Tapez la traduction en français
3. Valident et passent au suivant

💾 Les données sont stockées dans Google Sheets (cloud).

Usage:
    streamlit run crowdsource_app.py

Configuration (Streamlit Secrets):
    ADMIN_CODE=VotreCodeSecret
    GOOGLE_SPREADSHEET_ID=VotreIDSheet
    USE_GOOGLE_SHEETS=true

Auteur: Claude Opus 4.8
Date: 2026-06-06
"""

import streamlit as st
import os
import random
import sqlite3
import uuid
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
CV_PATH = PROJECT_ROOT / "data" / "cv-corpus-25.0-2026-03-09" / "gej"
CACHE_PATH = PROJECT_ROOT / "data" / ".mina_cache.db"

def get_setting(name, default=""):
    """Lit une configuration depuis l'environnement ou les secrets Streamlit."""

    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


# Configuration via secrets Streamlit ou environnement
USE_GOOGLE_SHEETS = get_setting("USE_GOOGLE_SHEETS", "false").lower() == "true"
GOOGLE_SPREADSHEET_ID = get_setting("GOOGLE_SPREADSHEET_ID")
ADMIN_CODE = get_setting("ADMIN_CODE")
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"


# =============================================================================
# FONCTIONS GOOGLE SHEETS
# =============================================================================

def init_google_sheets():
    """Initialise la connexion Google Sheets"""

    if not USE_GOOGLE_SHEETS:
        st.info("💡 Google Sheets désactivé - données en local uniquement")
        return None

    if not GOOGLE_SPREADSHEET_ID:
        st.warning("⚠️ GOOGLE_SPREADSHEET_ID non configuré")
        return None

    if not CREDENTIALS_FILE.exists():
        st.warning("⚠️ credentials.json non trouvé - Google Sheets indisponible")
        return None

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        credentials = Credentials.from_service_account_file(
            str(CREDENTIALS_FILE),
            scopes=scopes
        )
        gc = gspread.authorize(credentials)

        # Ouvrir le spreadsheet
        spreadsheet = gc.open_by_key(GOOGLE_SPREADSHEET_ID)
        sheet = spreadsheet.sheet1

        # Créer les en-têtes si vide
        if not sheet.get_all_records():
            sheet.append_row([
                'audio_id',
                'audio_path',
                'mina_text',
                'french_text',
                'translation_type',
                'created_at',
                'session_id'
            ])

        return sheet

    except Exception as e:
        st.warning(f"⚠️ Google Sheets: {e}")
        return None


def save_to_google_sheets(sheet, audio_id, mina_text, french_text, session_id):
    """Sauvegarde une traduction dans Google Sheets"""

    if sheet is None:
        return False

    try:
        from datetime import datetime

        sheet.append_row([
            audio_id,
            str(CV_PATH / 'clips' / f'{audio_id}.mp3'),
            mina_text,
            french_text,
            'text',
            datetime.now().isoformat(),
            session_id
        ])
        return True

    except Exception as e:
        st.error(f"Erreur Google Sheets: {e}")
        return False


def get_google_sheets_stats(sheet):
    """Récupère les stats depuis Google Sheets"""

    if sheet is None:
        return {'total': 0, 'unique': 0}

    try:
        records = sheet.get_all_records()
        total = len(records)

        # Compter les audios uniques
        seen = set()
        for r in records:
            seen.add(r.get('audio_id', ''))

        return {'total': total, 'unique': len(seen)}
    except:
        return {'total': 0, 'unique': 0}


# =============================================================================
# FONCTIONS BASE LOCALE (CACHE ONLY)
# =============================================================================

def init_local_cache():
    """Initialise le cache local SQLite (optionnel)"""

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(CACHE_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS translations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_id TEXT NOT NULL,
            audio_path TEXT,
            mina_text TEXT NOT NULL,
            french_text TEXT,
            translation_type TEXT DEFAULT 'text',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            synced INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS seen_audios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(audio_id, session_id)
        )
    """)

    conn.commit()
    conn.close()


def get_local_stats():
    """Récupère les statistiques du cache local."""

    init_local_cache()
    with sqlite3.connect(str(CACHE_PATH)) as conn:
        total = conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0]
        unique = conn.execute(
            "SELECT COUNT(DISTINCT audio_id) FROM translations"
        ).fetchone()[0]
    return {'total': total, 'unique': unique}


def save_to_local_cache(audio_id, mina_text, french_text, session_id):
    """Sauvegarde une traduction et marque l'audio comme vu."""

    init_local_cache()
    with sqlite3.connect(str(CACHE_PATH)) as conn:
        conn.execute(
            """
            INSERT INTO translations
                (audio_id, audio_path, mina_text, french_text, session_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (audio_id, str(CV_PATH / 'clips' / f'{audio_id}.mp3'),
             mina_text, french_text, session_id),
        )
        conn.execute(
            "INSERT OR IGNORE INTO seen_audios (audio_id, session_id) VALUES (?, ?)",
            (audio_id, session_id),
        )


def get_local_next_audio(audios, session_id):
    """Choisit un audio qui n'a pas encore été vu dans la session."""

    init_local_cache()
    with sqlite3.connect(str(CACHE_PATH)) as conn:
        rows = conn.execute(
            "SELECT audio_id FROM seen_audios WHERE session_id = ?",
            (session_id,),
        ).fetchall()
    seen_ids = {row[0] for row in rows}
    available = [audio for audio in audios if audio['id'] not in seen_ids]
    return random.choice(available) if available else None


def load_audio_list():
    """Charge la liste des audios Mina depuis Common Voice"""

    validated_tsv = CV_PATH / "validated.tsv"

    if not validated_tsv.exists():
        return []

    audios = []
    with open(validated_tsv, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]

        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                audio_path = parts[1]
                mina_text = parts[3]
                audio_id = audio_path.replace('.mp3', '')

                audios.append({
                    'id': audio_id,
                    'path': str(CV_PATH / 'clips' / audio_path),
                    'mina': mina_text
                })

    return audios


def get_next_audio(audios, session_id):
    """Récupère un audio aléatoire non encore vu (via Google Sheets)"""

    if not USE_GOOGLE_SHEETS or not GOOGLE_SPREADSHEET_ID:
        return get_local_next_audio(audios, session_id)

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        credentials = Credentials.from_service_account_file(
            str(CREDENTIALS_FILE),
            scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
        )
        gc = gspread.authorize(credentials)
        spreadsheet = gc.open_by_key(GOOGLE_SPREADSHEET_ID)
        sheet = spreadsheet.sheet1

        # Récupérer les audios déjà vus par cette session
        records = sheet.get_all_records()
        seen_ids = {r.get('audio_id', '') for r in records if r.get('session_id') == session_id}

        available = [a for a in audios if a['id'] not in seen_ids]

        if not available:
            return None

        return random.choice(available)

    except Exception:
        return get_local_next_audio(audios, session_id)


# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

def main():
    """Interface principale"""

    # Configuration
    st.set_page_config(
        page_title="Mina Crowdsource 🇹🇬",
        page_icon="🇹🇬",
        layout="centered"
    )

    # Style CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #1a1a2e;
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
    .stText, .stTextArea, p {
        color: #e0e0e0 !important;
    }
    .content-box {
        background-color: #16213e;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        border: 2px solid #0f3460;
    }
    .stButton > button {
        background-color: #e94560;
        color: white;
        font-size: 16px;
        padding: 12px 24px;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #ff6b6b;
    }
    [data-testid="stMetricValue"] {
        color: #00d9ff !important;
        font-size: 24px;
    }
    .info-box {
        background-color: #0f3460;
        padding: 15px;
        border-radius: 10px;
        color: #ffffff;
        margin: 10px 0;
    }
    .success-box {
        background-color: #1b5e20;
        padding: 15px;
        border-radius: 10px;
        color: #ffffff;
    }
    textarea {
        background-color: #0f3460 !important;
        color: #ffffff !important;
        border: 2px solid #00d9ff !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialisation session
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    if 'current_audio' not in st.session_state:
        st.session_state.current_audio = None

    # Initialiser Google Sheets
    google_sheet = init_google_sheets()

    # ============================================================
    # HEADER
    # ============================================================

    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #00d9ff; font-size: 40px; margin: 0;">🇹🇬 Mina Crowdsource</h1>
        <p style="color: #ffffff; font-size: 18px; margin: 10px 0;">
            Aidez-nous à traduire le <strong>Mina</strong> en <strong>Français</strong>
        </p>
        <p style="color: #888; font-size: 14px;">
            🎵 Écoutez • ✍️ Tapez la traduction • ✅ Validez
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Indicateur Google Sheets
    if google_sheet:
        st.markdown("""
        <div style="text-align: center; padding: 5px; margin-bottom: 15px;">
            <span style="background-color: #1b5e20; padding: 5px 15px; border-radius: 20px; color: #4caf50;">
                ☁️ Données sauvegardées dans Google Sheets ✅
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 5px; margin-bottom: 15px;">
            <span style="background-color: #f57c00; padding: 5px 15px; border-radius: 20px; color: #ffffff;">
                ⚠️ Mode local - Configurez Google Sheets pour la persistance
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ============================================================
    # STATISTIQUES
    # ============================================================

    stats = get_google_sheets_stats(google_sheet) if google_sheet else get_local_stats()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📚 Audios", "16,000+")

    with col2:
        st.metric("🌍 Total traduits", stats['total'])

    with col3:
        st.metric("👤 Votre session", st.session_state.session_id[:8])

    st.markdown("---")

    # ============================================================
    # SIDEBAR - ZONE ADMIN
    # ============================================================

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔐 Zone Admin")

        code_input = st.text_input("Entrez le code secret", type="password")

        if ADMIN_CODE and code_input == ADMIN_CODE:
            st.success("✅ Code correct!")

            st.markdown("---")
            st.markdown("### 📊 Statistiques")

            st.write(f"📝 Total: **{stats['total']}**")
            st.write(f"🎵 Uniques: **{stats['unique']}**")

            st.markdown("---")
            st.markdown("### 📋 Actions Admin")

            st.info("💡 Pour exporter les données:")
            st.markdown("""
            1. Ouvrez Google Sheets
            2. Fichier > Télécharger > CSV
            3. Convertissez en JSONL pour l'entraînement
            """)

        elif code_input:
            st.error("❌ Code incorrect")

    # ============================================================
    # ZONE AUDIO MINA
    # ============================================================

    audios = load_audio_list()

    if not audios:
        st.markdown("""
        <div class="info-box" style="text-align: center;">
            <h2 style="color: #ff6b6b;">❌ Dataset non trouvé</h2>
            <p>Le dataset Common Voice Mina doit être dans:</p>
            <code style="color: #00d9ff;">data/cv-corpus-25.0-2026-03-09/gej/</code>
        </div>
        """, unsafe_allow_html=True)
        return

    # Récupérer le prochain audio
    if st.session_state.current_audio is None:
        st.session_state.current_audio = get_next_audio(audios, st.session_state.session_id)

    audio = st.session_state.current_audio

    if audio is None:
        st.balloons()
        st.markdown("""
        <div style="text-align: center; padding: 40px;">
            <h2 style="color: #00d9ff;">🎉 Merci beaucoup !</h2>
            <p style="color: #ffffff; font-size: 18px;">
                Vous avez traduit tous les audios disponibles.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Zone audio
    st.markdown("""
    <div class="content-box">
        <h2 style="color: #00d9ff; text-align: center; margin-bottom: 20px;">
            🎵 Écoutez la phrase en Mina
        </h2>
    </div>
    """, unsafe_allow_html=True)

    audio_file = audio['path']

    if os.path.exists(audio_file):
        st.audio(audio_file, format="audio/mp3")
    else:
        st.markdown(f"""
        <div class="info-box" style="color: #ff6b6b;">
            ❌ Audio non trouvé: {audio_file}
        </div>
        """, unsafe_allow_html=True)
        return

    # Texte Mina de référence
    with st.expander("📝 Texte Mina original (clic pour voir)"):
        st.markdown(f"""
        <div style="background-color: #0f3460; padding: 15px; border-radius: 10px; color: #ffffff;">
            <strong>{audio['mina']}</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ============================================================
    # ZONE TRADUCTION
    # ============================================================

    st.markdown("""
    <div class="content-box">
        <h2 style="color: #00d9ff; text-align: center; margin-bottom: 20px;">
            ✍️ Tapez la traduction en Français
        </h2>
    </div>
    """, unsafe_allow_html=True)

    french_text = st.text_area(
        "",
        placeholder="Tapez ici votre traduction en français...",
        height=150,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # ============================================================
    # BOUTONS D'ACTION
    # ============================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        validate_clicked = st.button("✅ Valider", use_container_width=True)

    with col2:
        skip_clicked = st.button("⏭️ Passer", use_container_width=True)

    with col3:
        stop_clicked = st.button("🛑 Arrêter", use_container_width=True)

    # ============================================================
    # TRAITEMENT DES ACTIONS
    # ============================================================

    if validate_clicked:
        if french_text and len(french_text.strip()) > 0:
            # Sauvegarder localement, puis synchroniser vers Google Sheets.
            save_to_local_cache(
                audio_id=audio['id'],
                mina_text=audio['mina'],
                french_text=french_text.strip(),
                session_id=st.session_state.session_id
            )
            cloud_success = save_to_google_sheets(
                sheet=google_sheet,
                audio_id=audio['id'],
                mina_text=audio['mina'],
                french_text=french_text.strip(),
                session_id=st.session_state.session_id,
            )

            st.session_state.current_audio = get_next_audio(audios, st.session_state.session_id)

            if cloud_success:
                st.markdown("""
                <div class="success-box" style="text-align: center;">
                    <h3>✅ Traduction enregistrée dans Google Sheets !</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box" style="text-align: center;">
                    <h3>✅ Traduction enregistrée localement</h3>
                    <p>Google Sheets n'est pas configuré ou indisponible.</p>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="info-box" style="color: #ff6b6b; text-align: center;">
                ⚠️ Veuillez entrer une traduction
            </div>
            """, unsafe_allow_html=True)

        st.rerun()

    if skip_clicked:
        st.session_state.current_audio = get_next_audio(audios, st.session_state.session_id)
        st.rerun()

    if stop_clicked:
        st.balloons()
        st.markdown(f"""
        <div style="text-align: center; padding: 40px;">
            <h2 style="color: #00d9ff;">🎉 Merci pour votre participation !</h2>
            <p style="color: #888; font-size: 16px;">
                💡 Revenez demain pour continuer !
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ============================================================
    # FOOTER
    # ============================================================

    st.markdown("---")

    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #888;">
        <p>💡 Conseils :</p>
        <p>🎧 Écoutez l'audio plusieurs fois si nécessaire</p>
        <p>✍️ Traduisez de manière naturelle en français</p>
        <p>⏭️ Cliquez sur "Passer" si vous ne comprenez pas</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# EXÉCUTION
# =============================================================================

if __name__ == "__main__":
    main()