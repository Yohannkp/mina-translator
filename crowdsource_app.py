"""
crowdsource_app.py - Application Streamlit pour le crowdsourcing Mina → Français
================================================================================

Une application MINIMALISTE où les utilisateurs :
1. Écoutent une phrase en Mina (audio aléatoire)
2. Tapez la traduction en français
3. Valident et passent au suivant

Les données sont stockées dans SQLite (local) ET Google Sheets (cloud).

Usage:
    streamlit run crowdsource_app.py

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
DB_PATH = PROJECT_ROOT / "data" / "mina_crowdsource.db"
CV_PATH = PROJECT_ROOT / "data" / "cv-corpus-25.0-2026-03-09" / "gej"

# Google Sheets (optionnel - les données sont toujours stockées en SQLite)
USE_GOOGLE_SHEETS = os.environ.get("USE_GOOGLE_SHEETS", "false").lower() == "true"
GOOGLE_SPREADSHEET_ID = os.environ.get("GOOGLE_SPREADSHEET_ID", "")
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"

# Code secret pour télécharger la base (à changer!)
ADMIN_CODE = os.environ.get("Yohann", "1604")

# =============================================================================
# FONCTIONS GOOGLE SHEETS
# =============================================================================

def init_google_sheets():
    """Initialise la connexion Google Sheets"""

    if not USE_GOOGLE_SHEETS:
        return None

    if not GOOGLE_SPREADSHEET_ID:
        return None

    if not CREDENTIALS_FILE.exists():
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
        st.warning(f"⚠️ Google Sheets non disponible: {e}")
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
        return False


# =============================================================================
# FONCTIONS BASE DE DONNÉES
# =============================================================================

def init_database():
    """Initialise la base SQLite"""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
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
            session_id TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS seen_audios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_id TEXT NOT NULL UNIQUE,
            session_id TEXT,
            seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


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
    """Récupère un audio aléatoire non encore vu"""

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute(
        "SELECT audio_id FROM seen_audios WHERE session_id = ?",
        (session_id,)
    )
    seen_ids = {row[0] for row in cursor.fetchall()}
    conn.close()

    available = [a for a in audios if a['id'] not in seen_ids]

    if not available:
        return None

    return random.choice(available)


def save_translation(audio_id, mina_text, french_text, session_id, sheet=None):
    """Enregistre une traduction"""

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO translations
        (audio_id, audio_path, mina_text, french_text, translation_type, session_id)
        VALUES (?, ?, ?, ?, 'text', ?)
    """, (
        audio_id,
        str(CV_PATH / 'clips' / f'{audio_id}.mp3'),
        mina_text,
        french_text,
        session_id
    ))

    cursor.execute("""
        INSERT OR IGNORE INTO seen_audios (audio_id, session_id)
        VALUES (?, ?)
    """, (audio_id, session_id))

    conn.commit()
    conn.close()

    # Also save to Google Sheets if available
    if sheet:
        save_to_google_sheets(sheet, audio_id, mina_text, french_text, session_id)


def get_stats():
    """Récupère les statistiques"""

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM translations")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT audio_id) FROM translations")
    unique = cursor.fetchone()[0]

    conn.close()

    return {'total': total, 'unique': unique}


def get_user_stats(session_id):
    """Récupère les stats de l'utilisateur"""

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM seen_audios WHERE session_id = ?",
        (session_id,)
    )
    count = cursor.fetchone()[0]

    conn.close()

    return count


# =============================================================================
# INTERFACE STREAMLIT
# =============================================================================

def main():
    """Interface principale"""

    # Configuration
    st.set_page_config(
        page_title="Mina Crowdsource",
        page_icon="🇹🇬",
        layout="centered"
    )

    # Style CSS - fond sombre pour meilleur contraste
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
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
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
    textarea::placeholder {
        color: #888888 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialisation session
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    if 'current_audio' not in st.session_state:
        st.session_state.current_audio = None

    # Initialiser la base et Google Sheets
    init_database()
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
                ☁️ Synchronisé avec Google Sheets
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ============================================================
    # STATISTIQUES
    # ============================================================

    stats = get_stats()
    user_stats = get_user_stats(st.session_state.session_id)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📚 Audios", "16,000+")

    with col2:
        st.metric("🌍 Total traduits", stats['total'])

    with col3:
        st.metric("👤 Vos traductions", user_stats)

    st.markdown("---")

    # ============================================================
    # SIDEBAR - ZONE ADMIN (CODE SECRET)
    # ============================================================

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔐 Zone Admin")

        # Champ pour le code secret
        code_input = st.text_input("Entrez le code secret", type="password")

        if code_input == ADMIN_CODE:
            st.success("✅ Code correct!")

            # Bouton de téléchargement
            st.markdown("---")
            st.markdown("### 📥 Téléchargement")

            if os.path.exists(DB_PATH):
                with open(DB_PATH, "rb") as f:
                    db_data = f.read()
                    st.download_button(
                        label="💾 Télécharger la Base",
                        data=db_data,
                        file_name="mina_crowdsource.db",
                        mime="application/octet-stream"
                    )

                # Statistiques
                st.markdown("---")
                st.markdown("### 📊 Stats Base")
                stats = get_stats()
                st.write(f"Total traductions: **{stats['total']}**")
                st.write(f"Audios uniques: **{stats['unique']}**")
            else:
                st.warning("⚠️ Base non trouvée")

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
                Vous avez traduit tous les audios disponibles aujourd'hui.
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
    # ZONE TRADUCTION (TEXTE SEULEMENT)
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
            save_translation(
                audio_id=audio['id'],
                mina_text=audio['mina'],
                french_text=french_text.strip(),
                session_id=st.session_state.session_id,
                sheet=google_sheet
            )

            st.session_state.current_audio = get_next_audio(audios, st.session_state.session_id)

            st.markdown("""
            <div class="success-box" style="text-align: center;">
                <h3>✅ Traduction enregistrée ! Merci ! 🙏</h3>
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
            <p style="color: #ffffff; font-size: 18px;">
                Vous avez traduit <strong>{user_stats}</strong> phrases aujourd'hui.
            </p>
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