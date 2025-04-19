import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Téléchargement des stopwords (nécessaire si exécuté pour la première fois)
nltk.download("stopwords")

# Chargement du modèle et du vectorizer
model = joblib.load("data/sentiment_model.pkl")
vectorizer = joblib.load("data/tfidf_vectorizer.pkl")

# 🎨 Titre & sous-titre
st.set_page_config(page_title="Analyse de sentiment", page_icon="💬")
st.title("💬 Analyse de sentiment sur des avis clients")
st.caption("Modèle basé sur TF-IDF + Régression Logistique")

# 🧼 Fonction de nettoyage
def nettoyer(texte):
    stop_fr = set(stopwords.words("french"))
    texte = texte.lower()
    texte = re.sub(r"[^\w\s]", "", texte)
    mots = texte.split()
    mots = [mot for mot in mots if mot not in stop_fr]
    return " ".join(mots)

# 📝 Zone de saisie
texte_saisi = st.text_area("✍️ Saisis un avis client :", height=150)

# 🔍 Prédiction
if texte_saisi:
    texte_nettoye = nettoyer(texte_saisi)
    X = vectorizer.transform([texte_nettoye])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][prediction]

    # Affichage du résultat
    label = "✅ Positif" if prediction == 1 else "❌ Négatif"
    st.markdown("---")
    st.subheader("🔎 Résultat de l'analyse")
    st.markdown(f"**{label}** (confiance : `{proba*100:.2f}%`)")

    # Pour tester la sortie texte brute (facultatif)
    # st.text(f"Texte nettoyé : {texte_nettoye}")

