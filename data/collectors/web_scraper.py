"""
Web Scraper pour sources togolaises
Collecte de texte Mina/Français depuis sites web du Togo
"""
import requests
from bs4 import BeautifulSoup
import re
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import hashlib
import time

# Configuration
from config.settings import get_settings
settings = get_settings()


class TogoleseScraper:
    """Scraper pour sites togolais"""

    # Sources identifiées
    SOURCES = {
        # Sites gouvernementaux
        "republicoftogo": {
            "url": "https://www.republicoftogo.com",
            "selectors": ["article h2", "article p", ".content"],
            "lang": "fr",
        },
        "togolese_government": {
            "url": "https://www.gouv.tg",
            "selectors": ["p", "h2", "h3", ".article-content"],
            "lang": "fr",
        },
        # Médias togolais
        "togoonline": {
            "url": "https://www.togoactualite.com",
            "selectors": [".post-content", "article p", "h2"],
            "lang": "fr",
        },
        "togoforum": {
            "url": "https://www.togoforum.com",
            "selectors": ["[itemprop=articleBody]", ".content"],
            "lang": "fr",
        },
    }

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.CORPUS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        })
        self.collected = []

    def fetch_page(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Récupère le contenu d'une page"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response.text
            except Exception as e:
                logger.warning(f"Tentative {attempt+1} échouée pour {url}: {e}")
                time.sleep(2 ** attempt)
        return None

    def extract_text(self, html: str, selectors: List[str]) -> List[str]:
        """Extrait le texte via les selectors CSS"""
        soup = BeautifulSoup(html, "html.parser")
        texts = []
        for selector in selectors:
            try:
                elements = soup.select(selector)
                for el in elements:
                    text = el.get_text(strip=True)
                    if len(text) > 50:  # Filtrer texte trop court
                        texts.append(text)
            except Exception:
                continue
        return texts

    def scrape_site(self, site_name: str, pages: int = 10) -> List[Dict]:
        """Scrappe un site et ses pages"""
        if site_name not in self.SOURCES:
            logger.error(f"Site inconnu: {site_name}")
            return []

        config = self.SOURCES[site_name]
        base_url = config["url"]
        selectors = config["selectors"]
        lang = config["lang"]

        results = []
        logger.info(f"Scraping {base_url}")

        # Page principale
        html = self.fetch_page(base_url)
        if html:
            texts = self.extract_text(html, selectors)
            for text in texts:
                results.append({
                    "text": text,
                    "lang": lang,
                    "source": base_url,
                    "source_name": site_name,
                })

        # Pages supplémentaires (catégories, archives)
        additional_urls = [
            f"{base_url}/categories/news",
            f"{base_url}/actualites",
            f"{base_url}/news",
        ]

        for url in additional_urls[:pages]:
            html = self.fetch_page(url)
            if html:
                texts = self.extract_text(html, selectors)
                for text in texts:
                    results.append({
                        "text": text,
                        "lang": lang,
                        "source": url,
                        "source_name": site_name,
                    })
            time.sleep(1)  # Rate limiting

        logger.info(f"Collecté {len(results)} textes depuis {site_name}")
        return results

    def scrape_all(self, max_workers: int = 3) -> List[Dict]:
        """Scrappe toutes les sources configurées"""
        all_data = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.scrape_site, site): site
                for site in self.SOURCES.keys()
            }
            for future in as_completed(futures):
                site = futures[future]
                try:
                    results = future.result()
                    all_data.extend(results)
                except Exception as e:
                    logger.error(f"Erreur {site}: {e}")

        self.collected = all_data
        return all_data

    def save(self, filename: str = "scraped_data.json"):
        """Sauvegarde les données collectées"""
        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.collected, f, ensure_ascii=False, indent=2)

        logger.info(f"Sauvegardé {len(self.collected)} textes dans {output_path}")
        return output_path


def run_scraping():
    """Lance le scraping"""
    logger.info("Démarrage scraping sources togolaises")
    scraper = TogoleseScraper()
    data = scraper.scrape_all()
    scraper.save()
    return data


if __name__ == "__main__":
    run_scraping()