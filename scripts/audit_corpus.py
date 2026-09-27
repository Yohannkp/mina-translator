#!/usr/bin/env python3
"""Audit de qualite des corpus paralleles francais / mina.

Pourquoi ce script existe
-------------------------
Le mina (gen, ISO `gej`) n'a aucun corpus parallele public. Celui de ce projet a
donc ete constitue pour l'occasion, en partie par generation automatique. Une
donnee generee contient des defauts qu'on ne voit pas en la survolant : artefacts
du modele qui l'a produite, encodage casse, traductions dupliquees.

Ce script mesure ces defauts au lieu de les supposer. Il ne juge pas si une
traduction est correcte -- cela demanderait un locuteur. Il ne detecte que des
anomalies structurelles, verifiables sans connaitre la langue.

Usage
-----
    python scripts/audit_corpus.py                 # tous les corpus
    python scripts/audit_corpus.py --json          # sortie machine
    python scripts/audit_corpus.py --seuil 5       # echoue si > 5 % d'anomalies
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
CORPUS_DIR = RACINE / "data" / "corpus"

# Les schemas different d'un fichier a l'autre : on normalise a (source_fr, cible_mina).
CLES_FR = ("french", "fr", "input")
CLES_MINA = ("mina", "output", "target")

# Traces laissees par un LLM qui a produit la traduction au lieu de traduire.
ARTEFACTS_LLM = (
    "je ne comprends pas",
    "la réponse est",
    "la reponse est",
    "je suis désolé",
    "je suis desole",
    "en tant que",
    "voici la traduction",
    "il semble que",
    "peut-être une expression",
)

# L'alphabet du gen/ewe. Il comprend des lettres absentes du latin de base :
# ɖ, ƒ, ŋ, ʋ et surtout ɣ (fricative velaire voisee), plus les voyelles
# ouvertes ɔ et ɛ. Les variantes ə/ǝ et ẹ apparaissent dans certaines
# orthographies gbe : on les accepte plutot que de les signaler a tort.
ALPHABET_ATTENDU = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "ɔƆɛƐɖƉƒƑŋŊʋƲɣƔ"
    "əƏǝẹẸọỌ"
    "ãẽĩõũáàâéèêíìîóòôúùûç"
    "\u0300\u0301\u0303\u0304\u0331"
    "0123456789"
)

# Deux facons d'ecrire le meme phoneme : signe d'une provenance heterogene.
GRAPHIES_CONCURRENTES = {"o ouvert": ("ɔ", "ᴐ"), "e ouvert": ("ɛ", "ɜ")}


def paires(chemin: Path) -> list[tuple[str, str]]:
    """Lit un .jsonl et renvoie les couples (francais, mina), quel que soit le schema."""
    sorties = []
    with chemin.open(encoding="utf-8", errors="replace") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            try:
                obj = json.loads(ligne)
            except json.JSONDecodeError:
                sorties.append(("<ligne json invalide>", ""))
                continue
            fr = next((obj[k] for k in CLES_FR if obj.get(k)), "")
            mi = next((obj[k] for k in CLES_MINA if obj.get(k)), "")
            sorties.append((str(fr), str(mi)))
    return sorties


def hors_alphabet(texte: str) -> list[str]:
    """Caracteres qui n'appartiennent ni a l'alphabet gen ni a la ponctuation."""
    anormaux = []
    for c in texte:
        if c in ALPHABET_ATTENDU or c.isspace():
            continue
        cat = unicodedata.category(c)
        if cat.startswith(("P", "Z", "M")):  # ponctuation, espaces, diacritiques
            continue
        anormaux.append(c)
    return anormaux


def recouvrement(source: str, cible: str) -> float:
    """Part des mots de la cible qui figurent deja dans la source.

    Une traduction authentique est proche de zero : deux langues ne partagent
    pas leur vocabulaire. Une valeur elevee signale une source recopiee au lieu
    d'etre traduite. Les mots capitalises sont ignores : un nom propre se
    retrouve legitimement des deux cotes.
    """
    def mots(t: str) -> list[str]:
        return [m.lower() for m in re.findall(r"[^\W\d_]+", t, re.UNICODE) if not m[:1].isupper()]

    cibles, sources = mots(cible), set(mots(source))
    if len(cibles) < 3:
        return 0.0
    return sum(1 for m in cibles if m in sources) / len(cibles)


def auditer(chemin: Path) -> dict:
    lignes = paires(chemin)
    total = len(lignes)
    if total == 0:
        return {"fichier": chemin.name, "total": 0, "ignore": "fichier vide"}

    # Tous les fichiers de data/corpus ne sont pas des corpus paralleles :
    # mina_full_dataset.jsonl, par exemple, est une liste de transcriptions
    # Common Voice sans traduction. Les auditer comme des paires n'a pas de sens.
    apparies = sum(1 for fr, mi in lignes if fr.strip() and mi.strip())
    if apparies < 0.5 * total:
        return {"fichier": chemin.name, "total": total,
                "ignore": "pas un corpus parallele (aucune colonne source/cible exploitable)"}

    anomalies: dict[str, list[int]] = defaultdict(list)
    par_cible: dict[str, list[int]] = defaultdict(list)

    for i, (fr, mi) in enumerate(lignes):
        bas = fr.lower()

        if "Ã" in fr or "Ã" in mi:
            anomalies["encodage casse (mojibake)"].append(i)
        if any(a in bas for a in ARTEFACTS_LLM):
            anomalies["artefact de generation (le LLM commente au lieu de traduire)"].append(i)
        if len(fr) > 90 and not fr.rstrip().endswith((".", "!", "?", '"', "»")):
            anomalies["source tronquee"].append(i)
        if not fr.strip() or not mi.strip():
            anomalies["cote vide"].append(i)
        if fr.strip() and fr.strip().lower() == mi.strip().lower():
            anomalies["source recopiee en cible"].append(i)
        elif recouvrement(fr, mi) >= 0.4:
            anomalies["source partiellement recopiee en cible"].append(i)
        etrangers = hors_alphabet(mi)
        if etrangers:
            anomalies[f"caractere hors alphabet ({''.join(sorted(set(etrangers)))})"].append(i)
        if mi.strip():
            par_cible[mi.strip()].append(i)

    # Une meme traduction pour des sources differentes : impossible en traduction humaine.
    for cible, idx in par_cible.items():
        sources = {lignes[i][0].strip().lower() for i in idx}
        if len(sources) > 1:
            anomalies["meme traduction pour des sens differents"].extend(idx)

    graphies = {}
    for nom, variantes in GRAPHIES_CONCURRENTES.items():
        compte = {v: sum(1 for _, mi in lignes if v in mi) for v in variantes}
        presentes = {v: n for v, n in compte.items() if n}
        if len(presentes) > 1:
            graphies[nom] = presentes

    touchees = set()
    for idx in anomalies.values():
        touchees.update(idx)

    return {
        "fichier": chemin.name,
        "total": total,
        "lignes_touchees": len(touchees),
        "part": round(100 * len(touchees) / total, 1),
        "anomalies": {k: len(set(v)) for k, v in sorted(anomalies.items(), key=lambda x: -len(set(x[1])))},
        "graphies_concurrentes": graphies,
        "exemples": [
            {"ligne": i + 1, "fr": lignes[i][0][:90], "mina": lignes[i][1][:60], "motif": motif}
            for motif, idx in list(anomalies.items())[:4]
            for i in sorted(set(idx))[:1]
        ],
    }


def principal() -> int:
    ap = argparse.ArgumentParser(description="Audit de qualite des corpus paralleles.")
    ap.add_argument("fichiers", nargs="*", help="fichiers .jsonl (defaut : tous ceux de data/corpus)")
    ap.add_argument("--json", action="store_true", help="sortie JSON")
    ap.add_argument("--seuil", type=float, default=None,
                    help="part maximale d'anomalies toleree, en %% ; au-dela le script echoue")
    args = ap.parse_args()

    chemins = [Path(f) for f in args.fichiers] or sorted(CORPUS_DIR.glob("*.jsonl"))
    if not chemins:
        print(f"Aucun corpus trouve dans {CORPUS_DIR}", file=sys.stderr)
        return 2

    rapports = [auditer(p) for p in chemins if p.exists()]

    if args.json:
        print(json.dumps(rapports, ensure_ascii=False, indent=2))
    else:
        print(f"\nAudit des corpus — {len(rapports)} fichier(s)\n" + "=" * 78)
        for r in rapports:
            if r.get("ignore"):
                print(f"\n{r['fichier']}  —  ecarte : {r['ignore']}")
                continue
            if not r["total"]:
                continue
            print(f"\n{r['fichier']}  —  {r['total']} paires, "
                  f"{r['lignes_touchees']} touchees ({r['part']} %)")
            if not r["anomalies"]:
                print("   aucune anomalie structurelle detectee")
            for motif, n in r["anomalies"].items():
                print(f"   {n:>5}  {motif}")
            for nom, compte in r["graphies_concurrentes"].items():
                detail = ", ".join(f"« {v} » dans {n}" for v, n in compte.items())
                print(f"         graphies concurrentes pour le {nom} : {detail}")
            for ex in r["exemples"]:
                print(f"         ex. l.{ex['ligne']} [{ex['motif']}]  {ex['fr']} -> {ex['mina']}")
        print("\n" + "=" * 78)
        print("Ce script ne juge pas la justesse des traductions : cela demande un locuteur.")
        print("Il ne detecte que des anomalies verifiables sans connaitre la langue.\n")

    if args.seuil is not None:
        pires = [r for r in rapports if not r.get("ignore") and r.get("part", 0) > args.seuil]
        if pires:
            noms = ", ".join(r["fichier"] for r in pires)
            print(f"ECHEC : {noms} depasse le seuil de {args.seuil} %", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())
