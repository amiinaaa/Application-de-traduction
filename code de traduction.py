from transformers import MBartForConditionalGeneration, MBart50Tokenizer

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

# Exemple d'utilisation
source_text = "Bonjour, comment allez-vous?"
source_lang = "fr_XX"  # Français
target_lang = "en_XX"  # Anglais

translated_text = translate(source_text, source_lang, target_lang)
print(f"Source: {source_text}")
print(f"Translated: {translated_text}")

# Entrée utilisateur
source_text = input("Entrez le texte à traduire : ")
source_lang = input("Entrez la langue source (code de langue mBART, ex: 'fr_XX' pour Français) : ")
target_lang = input("Entrez la langue cible (code de langue mBART, ex: 'en_XX' pour Anglais) : ")

translated_text = translate(source_text, source_lang, target_lang)
print(f"Source: {source_text}")
print(f"Translated: {translated_text}")