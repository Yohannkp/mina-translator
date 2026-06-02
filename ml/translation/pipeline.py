"""
Pipeline de Traduction Français <-> Mina
Version basique: utilise l prompting + modèle multilingue
Version avancée: QLoRA fine-tuned sur corpus Mina
"""
from typing import Optional, Literal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from loguru import logger


class TranslationPipeline:
    """
    Pipeline de traduction FR <-> Mina
    Utilise un LLM multilingue avec prompting
    """

    SYSTEM_PROMPT_FR_TO_MINA = """Tu es un traducteur expert français-mina.
Traduis ONLY le texte suivant du français vers le mina (langue du Togo).
Ne translates PAS les noms propres.
Réponds UNIQUEMENT avec la traduction, sans explications.

Exemple:
Français: Bonjour, comment allez-vous?
Mina: Mɛlɔ, ntsɛ nyabi?

Maintenant traduis:
Français: {text}
Mina:"""

    SYSTEM_PROMPT_MINA_TO_FR = """You are an expert Mina-French translator.
Translate ONLY the following text from Mina to French.
Do NOT translate proper names.
Respond ONLY with the translation, without explanations.

Example:
Mina: Mɛlɔ, ntsɛ nyabi?
French: Bonjour, comment allez-vous?

Now translate:
Mina: {text}
French:"""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        load_in_4bit: bool = True,
        device: Optional[str] = None,
    ):
        self.model_name = model_name

        # Device detection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Chargement modèle de traduction: {model_name}")
        logger.info(f"Device: {self.device}, 4-bit: {load_in_4bit}")

        # Quantification pour 8GB VRAM
        if load_in_4bit and self.device == "cuda":
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            self.quantization_config = None

        # Chargement tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Chargement modèle
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        # Pipeline HF
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            do_sample=False,  # Greedy pour traduction déterministe
            pad_token_id=self.tokenizer.pad_token_id,
        )

        logger.info("Pipeline traduction prêt!")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Traduit un texte entre français et mina.

        Args:
            text: Texte à traduire
            source_lang: "fr" ou "mina"
            target_lang: "fr" ou "mina"

        Returns:
            Texte traduit
        """
        if source_lang == target_lang:
            return text

        # Construction du prompt
        if source_lang == "fr" and target_lang == "mina":
            system_prompt = self.SYSTEM_PROMPT_FR_TO_MINA.format(text=text)
        elif source_lang == "mina" and target_lang == "fr":
            system_prompt = self.SYSTEM_PROMPT_MINA_TO_FR.format(text=text)
        else:
            raise ValueError(f"Combinaison non supportée: {source_lang} -> {target_lang}")

        # Génération
        outputs = self.pipeline(system_prompt)
        translation = outputs[0]["generated_text"]

        # Extraction du texte traduit (après le marker ":")
        if ":\n" in translation:
            parts = translation.split(":\n")
            if len(parts) >= 2:
                # Prendre la dernière partie (la traduction)
                translation = parts[-1].strip()
            else:
                translation = translation.split(":\n")[1].strip()

        # Nettoyage final
        translation = translation.strip()

        logger.debug(f"Traduit: {text[:50]}... -> {translation[:50]}...")

        return translation

    def translate_to_mina(self, text: str) -> str:
        """Traduit du français vers le mina"""
        return self.translate(text, source_lang="fr", target_lang="mina")

    def translate_to_french(self, text: str) -> str:
        """Traduit du mina vers le français"""
        return self.translate(text, source_lang="mina", target_lang="fr")

    def get_vram_usage(self) -> dict:
        """Retourne l'utilisation VRAM actuelle"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
            }
        return {"allocated_gb": 0, "reserved_gb": 0}


def load_translation_pipeline(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    load_in_4bit: bool = True,
) -> TranslationPipeline:
    """Factory function pour charger le pipeline de traduction"""
    return TranslationPipeline(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
    )


if __name__ == "__main__":
    logger.info("Test du pipeline traduction")
    pipeline = load_translation_pipeline()

    # Test
    test_text = "Bonjour, comment allez-vous?"
    result = pipeline.translate_to_mina(test_text)
    print(f"FR -> MINA: {result}")

    logger.info("Translation Pipeline prêt!")