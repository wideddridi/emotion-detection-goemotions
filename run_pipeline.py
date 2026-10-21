"""
run_pipeline.py
---------------
Script d'exécution du pipeline complet de détection d'émotions.

Usage :
    python run_pipeline.py

Prérequis :
    - Modèle entraîné sauvegardé dans  models/best_svm.joblib
    - Vectoriseur sauvegardé dans       models/tfidf_vectorizer.joblib
    - Fichier de labels dans            data/raw/emotions.txt

Pour entraîner le modèle depuis zéro, exécuter les notebooks dans l'ordre :
    notebooks/01_data_exploration.ipynb
    notebooks/02_preprocessing.ipynb
    notebooks/03_model_training.ipynb
"""

import os
import joblib
import pandas as pd

from src.text_cleaner import preprocess_text


# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "models", "best_svm.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.joblib")
EMOTIONS_FILE   = os.path.join(BASE_DIR, "data", "raw", "emotions.txt")


def load_emotion_mapping(path: str) -> dict:
    """Charge le fichier emotions.txt et retourne un dict {index: nom}."""
    mapping = {}
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            name = line.strip()
            if name:
                mapping[idx] = name
    return mapping


def predict_emotion(texts: list, model, vectorizer, emotion_mapping: dict) -> pd.DataFrame:
    """
    Prédit l'émotion de nouveaux textes.

    Paramètres :
        texts          : liste de strings (textes bruts anglais)
        model          : modèle entraîné (doit avoir .predict())
        vectorizer     : TfidfVectorizer déjà fitté sur le train
        emotion_mapping: dict {label_int: nom_émotion}

    Retourne :
        DataFrame : texte original, texte nettoyé, label, émotion prédite
    """
    cleaned = [preprocess_text(t) for t in texts]
    X       = vectorizer.transform(cleaned)
    preds   = model.predict(X)

    return pd.DataFrame({
        "Texte Original" : texts,
        "Texte Nettoyé"  : cleaned,
        "Label"          : preds,
        "Émotion Prédite": [emotion_mapping.get(p, str(p)) for p in preds],
    })


def main():
    # ── Chargement des artefacts ──────────────────────────────────────────────
    print("⏳ Chargement du modèle et du vectoriseur…")
    model          = joblib.load(MODEL_PATH)
    vectorizer     = joblib.load(VECTORIZER_PATH)
    emotion_mapping = load_emotion_mapping(EMOTIONS_FILE)
    print("✅ Artefacts chargés.")

    # ── Exemples de prédiction ────────────────────────────────────────────────
    new_texts = [
        "I absolutely love this! It made my day so much better!",
        "This is the worst experience ever. I'm furious at this system!",
        "The package arrived today. It contains 3 items.",
        "I'm so disappointed, I was really hoping for more.",
        "Wow, this is incredibly awesome! Thank you so much!",
        "I feel terrible about what happened, I regret everything.",
        "I hate waiting in long lines, it makes me so angry.",
        "Haha that's actually really funny, I can't stop laughing!",
        "I'm scared of what might happen next.",
        "The weather is nice today, nothing special.",
    ]

    predictions = predict_emotion(new_texts, model, vectorizer, emotion_mapping)

    print("\n🎭 PRÉDICTIONS SUR DE NOUVEAUX TEXTES :")
    print("═" * 72)
    for _, row in predictions.iterrows():
        print(f"  {row['Émotion Prédite']:<18} | {row['Texte Original'][:55]}")
    print("═" * 72)


if __name__ == "__main__":
    main()
