"""
text_cleaner.py
---------------
Fonctions de nettoyage de texte pour le projet GoEmotions.
Extraites depuis le notebook emotion_detection.ipynb — Section 4 (Prétraitement).
"""

import re


def preprocess_text(text: str) -> str:
    """
    Nettoie un texte anglais pour le NLP.

    Paramètre : text (str) — texte brut
    Retourne  : str       — texte nettoyé

    Étapes :
        1. Mise en minuscules
        2. Expansion des contractions (won't → will not, etc.)
        3. Suppression des URLs
        4. Suppression des mentions / références Reddit (@user, #tag, r/sub, u/user)
        5. Suppression des chiffres
        6. Suppression de la ponctuation (apostrophes conservées)
        7. Normalisation des espaces
    """
    # 1. Minuscules + uniformisation des apostrophes typographiques
    text = text.lower()
    text = text.replace("\u2019", "'")

    contractions = {
        "won't": "will not", "can't": "cannot", "don't": "do not",
        "it's": "it is", "i'm": "i am", "you're": "you are",
        "they're": "they are", "we're": "we are", "that's": "that is",
        "what's": "what is", "who's": "who is", "there's": "there is",
        "here's": "here is", "he's": "he is", "she's": "she is",
        "how's": "how is", "where's": "where is", "when's": "when is",
        # Suffixes génériques — ordre important : plus spécifiques en premier
        "n't": " not", "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'ve": " have", "'m": " am",
    }
    for contraction, expansion in contractions.items():
        text = re.sub(contraction, expansion, text)

    # 2. Suppression des URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # 3. Suppression mentions et références Reddit
    text = re.sub(r'@\w+|#\w+|r/\w+|u/\w+', '', text)
    # 4. Suppression des chiffres
    text = re.sub(r'\d+', '', text)
    # 5. Suppression ponctuation (apostrophes conservées)
    text = re.sub(r"[^a-z\s']", ' ', text)
    text = re.sub(r'-+', ' ', text)
    # 6. Normalisation des espaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ── Bloc de test rapide ───────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "I am so ANGRY at this stupid system!!! http://example.com",
        "This is the best day ever, I'm so happy! Thanks @user #blessed",
        "r/python has 1000 tips for beginners, loved reading it!",
        "I am NOT happy at all about this decision",
        "I liked i'm the idea but I felt really disappointed",
    ]
    print("📝 Tests de nettoyage :")
    print("-" * 62)
    for t in tests:
        print(f"AVANT  : {t}")
        print(f"APRÈS  : {preprocess_text(t)}")
        print("-" * 62)
