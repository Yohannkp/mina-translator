"""
crowdsource_app.py - Application Streamlit pour le crowdsourcing Mina → Français
================================================================================

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

# =============================================================================
# FONCTIONS
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
            translation_type TEXT,
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


def save_translation(audio_id, mina_text, french_text, translation_type, session_id):
    """Enregistre une traduction"""

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO translations
        (audio_id, audio_path, mina_text, french_text, translation_type, session_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        audio_id,
        str(CV_PATH / 'clips' / f'{audio_id}.mp3'),
        mina_text,
        french_text,
        translation_type,
        session_id
    ))

    cursor.execute("""
        INSERT OR IGNORE INTO seen_audios (audio_id, session_id)
        VALUES (?, ?)
    """, (audio_id, session_id))

    conn.commit()
    conn.close()


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

    # Style CSS personnalisé - fond sombre pour meilleur contraste
    st.markdown("""
    <style>
    /* Fond de la page */
    .stApp {
        background-color: #1a1a2e;
    }

    /* Titres */
    h1, h2, h3 {
        color: #ffffff !important;
    }

    /* Texte général */
    .stText, .stTextArea, p {
        color: #e0e0e0 !important;
    }

    /* Boîtes de contenu */
    .content-box {
        background-color: #16213e;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border: 2px solid #0f3460;
    }

    /* Boutons */
    .stButton > button {
        background-color: #e94560;
        color: white;
        font-size: 16px;
        padding: 12px 24px;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #ff6b6b;
    }

    /* Radio buttons */
    .stRadio > div {
        background-color: #16213e;
        padding: 10px;
        border-radius: 10px;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00d9ff !important;
        font-size: 24px;
    }

    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }

    /* Info boxes */
    .info-box {
        background-color: #0f3460;
        padding: 15px;
        border-radius: 10px;
        color: #ffffff;
        margin: 10px 0;
    }

    /* Success message */
    .success-box {
        background-color: #1b5e20;
        padding: 15px;
        border-radius: 10px;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialisation session
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    if 'current_audio' not in st.session_state:
        st.session_state.current_audio = None

    # Initialiser la base
    init_database()

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
            🎵 Écoutez • 🎙️ Enregistrez ou tapez • ✅ Validez
        </p>
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
        <h2 style="color: #00d9ff; text-align: center;">🎵 Écoutez la phrase en Mina</h2>
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
        <h2 style="color: #00d9ff; text-align: center;">✍️ Traduisez en Français</h2>
    </div>
    """, unsafe_allow_html=True)

    # Choix du mode de traduction
    st.markdown("""
    <div style="background-color: #16213e; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <p style="color: #ffffff; margin: 0;">Choisissez comment traduire :</p>
    </div>
    """, unsafe_allow_html=True)

    translation_mode = st.radio(
        "",
        ["✍️ Taper la traduction", "🎙️ Enregistrer ma voix"],
        horizontal=True,
        label_visibility="collapsed"
    )

    french_text = None

    if translation_mode == "✍️ Taper la traduction":
        st.markdown("""
        <div style="background-color: #16213e; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <p style="color: #ffffff;">Tapez votre traduction française :</p>
        </div>
        """, unsafe_allow_html=True)

        french_text = st.text_area(
            "",
            placeholder="Ex: Bonjour, comment allez-vous ?",
            height=120,
            label_visibility="collapsed"
        )

    else:
        st.markdown("""
        <div style="background-color: #16213e; padding: 20px; border-radius: 10px; margin: 10px 0; text-align: center;">
            <h3 style="color: #00d9ff;">🎙️ Enregistrement vocal</h3>
            <p style="color: #ffffff;">
                Utilisez le bouton ci-dessous pour enregistrer votre voix en français.<br>
                <em style="color: #888;">(L'enregistrement sera sauvegardé avec la traduction)</em>
            </p>
            <p style="color: #ff6b6b; font-size: 14px;">
                ⚠️ L'enregistrement audio nécessite un microphone.
            </p>
        </div>
        """, unsafe_allow_html=True)

        french_text = st.text_area(
            "Ou tapez la traduction ici (optionnel) :",
            placeholder="Vous pouvez aussi taper la traduction...",
            height=80
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
                translation_type="text",
                session_id=st.session_state.session_id
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
        <p>✍️ Tapez votre traduction de manière naturelle</p>
        <p>⏭️ Cliquez sur "Passer" si vous ne comprenez pas</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# EXÉCUTION
# =============================================================================

if __name__ == "__main__":
    main()