from deep_translator import GoogleTranslator
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class TranslationService:
    """Handles translation between pt-BR and en-US"""

    def __init__(self):
        self.pt_to_en = GoogleTranslator(source='pt', target='en')
        self.en_to_pt = GoogleTranslator(source='en', target='pt')

    def translate(self, text: str, source_lang: str) -> Tuple[str, str]:
        """
        Translate text bidirectionally based on detected source language.

        Args:
            text: Text to translate
            source_lang: Source language code ('pt' or 'en')

        Returns:
            Tuple of (original_text, translated_text)
        """
        if not text or not text.strip():
            return text, ""

        try:
            # Normalize language code
            if source_lang.startswith('pt'):
                # Portuguese to English
                translated = self.pt_to_en.translate(text)
                return text, translated
            else:
                # English to Portuguese (default)
                translated = self.en_to_pt.translate(text)
                return text, translated

        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Return original text if translation fails
            return text, text

    def get_target_language(self, source_lang: str) -> str:
        """Get target language code based on source"""
        if source_lang.startswith('pt'):
            return 'en'
        return 'pt'


# Singleton instance
translation_service = TranslationService()
