from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from gtts import gTTS
import speech_recognition as sr
import os

# Charger le modèle et le tokenizer pour la traduction
model_name = 'facebook/mbart-large-50-many-to-many-mmt'
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

def translate(text, src_lang, tgt_lang):
    # Définir la langue source et cible
    tokenizer.src_lang = src_lang
    encoded_text = tokenizer(text, return_tensors="pt")

    # Générer la traduction
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])

    # Décoder le texte traduit
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

def text_to_speech(text, lang):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("output.mp3")
    os.system("start output.mp3")  # Sur Windows, utilisez "start". Pour MacOS, utilisez "afplay". Pour Linux, utilisez "mpg321".

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Parlez maintenant...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language="fr-FR")  # Changez "fr-FR" par le code de langue souhaité
        print(f"Vous avez dit : {text}")
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition n'a pas pu comprendre l'audio")
        return ""
    except sr.RequestError as e:
        print(f"Erreur de service de Google Speech Recognition; {e}")
        return ""

# Entrée utilisateur via reconnaissance vocale
source_text = speech_to_text()
source_lang = input("Entrez la langue source (code de langue mBART, ex: 'fr_XX' pour Français) : ")
target_lang = input("Entrez la langue cible (code de langue mBART, ex: 'en_XX' pour Anglais) : ")

# Traduction
translated_text = translate(source_text, source_lang, target_lang)
print(f"Source: {source_text}")
print(f"Translated: {translated_text}")

# Synthèse vocale
text_to_speech(source_text, source_lang[:2])  # Lire la phrase source
text_to_speech(translated_text, target_lang[:2])  # Lire la phrase traduite