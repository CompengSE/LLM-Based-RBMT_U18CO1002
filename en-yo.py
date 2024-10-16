import json
import google.generativeai as genai
import os
import nltk
import spacy
from nltk.translate.bleu_score import sentence_bleu
import random

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

nltk_data_dir = './'  # Replace with your desired path
nltk.data.path.append(nltk_data_dir)

# Download specific resources to the specified directory
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)

def load_json_file(json_file_path):

    with open(json_file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)  # Load the JSON data
    return data


# read stored bilingual dictionary

storedPath = './bi-Dictionary.json'
wordPairs = load_json_file(storedPath)


def getWordpair(word: str) -> str:
    return wordPairs.get(word, word)


os.environ["API_KEY"] = '<Gemini_API_KEY_HERE>'
genai.configure(api_key=os.environ["API_KEY"])

def make_prompt(query):

    # Crafting the prompt for Gemini

    prompt = (
        "You are an AI bot designed to transform sentences into the SVO (Subject-Verb-Object) structure. "
        "Ensure that all punctuation provided in the input is included in the output and placed correctly.\n\n"
        
        "The length of the input list of POS-tagged words should match the length of the output list.\n\n"
        
        "Follow these grammar rules for Yoruba translation:\n"

        "Translation Guidelines:\n"
        "   - Begin by analyzing the verb in the English sentence to determine if it's transitive or intransitive.\n"
        "   - Use the phrase structure rules above to guide the translation:\n"
        "      - For transitive sentences, follow: S → NP (Aux) VP NP.\n"
        "      - For intransitive sentences, use: S → NP (Aux) VP.\n"
        "      - For verbs that don't require an object, follow: S → NP (Aux) VP, with optional adverbial phrases: VP → V AdvP.\n\n"
        
        
        "Verb Phrase (VP) Rules:\n"
        "   - VP → V: A simple verb phrase.\n"
        "   - VP → V NP: A verb followed by a noun phrase.\n"
        "   - VP → V N ADJ: A verb followed by a noun and an adjective.\n"
        "   - VP → V N ADJ DET PP: A verb followed by a noun, adjective, determiner, and prepositional phrase.\n"
        "   - VP → V ADJ N PP: A verb followed by an adjective, noun, and prepositional phrase.\n"
        "   - VP → V DET ADJ N PP: A verb followed by a determiner, adjective, noun, and prepositional phrase.\n"
        "   - VP → V DET ADJ N P NP: A verb followed by a determiner, adjective, noun, and a prepositional phrase with a noun phrase.\n"
        "   - VP → V DET ADJ N P DET N: A verb followed by a determiner, adjective, noun, preposition, determiner, and noun.\n\n"
        
        "Noun Phrase (NP) Rules:\n"
        "   - NP → N: A noun phrase consisting of a single noun.\n"
        "   - NP → DET N: A noun phrase with a determiner and a noun.\n"
        "   - NP → DET ADJ N: A noun phrase with a determiner, adjective, and noun.\n\n"
        
        "Prepositional Phrase (PP) Rules:\n"
        "   - PP → P NP: A prepositional phrase with a preposition followed by a noun phrase.\n\n"
        
        "Make sure to handle adjectives, determiners, and prepositions according to the specified phrase structure rules.\n\n"
        
        "If the resulting sentence ends with a Determiner (POS tag: 'DT'), remove the Determiner to ensure grammatical correctness.\n\n"
        
        "Examples:\n"
        f"   - SENTENCE TO CONVERT: {nltk.pos_tag(nltk.word_tokenize('the boy eat the apple'))}\n"
        "     ai_result: [\"boy\", \"the\", \"eat\", \"apple\"]\n"
        f"   - SENTENCE TO CONVERT: '{nltk.pos_tag(nltk.word_tokenize('Who is the author of this article'))}'\n"
        "     ai_result: [\"who\", \"is\", \"the\", \"article\", \"author\", \"of\"]\n\n"
    
        "Format the response as:\n"
        "{\n"
        '  "ai_result": "<list of tokenized SVO-formatted words enclosed in double quotes>",\n'
        "}\n\n"
        
        f"SENTENCE TO CONVERT: '{query}'\n"
    )

    return prompt


config = genai.GenerationConfig(
    max_output_tokens=2048, temperature=0.55, top_p=0.8, top_k=22
)


def translate_sentence(sentence: str):

    words = nltk.word_tokenize(sentence)

    # Perform POS tagging
    pos_tags = nltk.pos_tag(words)
    # print("\n POS: ")
    # print(pos_tags)
    
    
    # Call to AI
    
    prompt = make_prompt(pos_tags)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(prompt, generation_config=config)
    try:
        response = response.text.replace('`', '').replace('json', '')
        data = json.loads(response)
        
        # Printing the parsed JSON
        print(data['ai_result'])
        
        result = " ".join([getWordpair(word.lower()) for word in data['ai_result']])
        return result
    except:
        return ""



querys = ["Who is the author of this article?","my name is good","ade killed a goat",  "The lazy fox jumped over the lazy dog"]
sen2 = "the boy eat the apple"

print()
print("without word re odering: ")
print("English sentence: ", sen2)
print("Yoruba translation: ", " ".join([wordPairs[word.lower()] for word in sen2.split(" ")]))
print()
print("With word re ordering: ")
print("Yoruba translation: ", translate_sentence(sen2))


for sent in querys:
    translate_sentence(sent)    



# Download test Data
storePath = './testData.json'
df_test = load_json_file(storePath)

sentences = random.sample(df_test, 50)

test1 = sentences[0]

candidate = translate_sentence(test1['en'])
print(candidate, " --- ", test1['yo'])
print(f"BLEU SCORE: {sentence_bleu([test1['yo'].lower().split()], candidate.split(), weights=(1, 0, 0, 0))}")


from sacrebleu import sentence_bleu as sb
import time

print(f"Sacrebleu SCORE: {sb(candidate, [test1['yo'].lower()]).score}")

total_bleu_score = 0
total_sacred_score = 0

for i in range(len(sentences)):
    sentence = sentences[i]
    candidate = translate_sentence(sentence['en'])
    target  = sentence['yo']
    total_bleu_score += sentence_bleu([target.lower().split()], candidate.split(), weights=(1, 0, 0, 0))
    total_sacred_score += sb(candidate, [target.lower()]).score
    if i > 0 :
        print(f"Running average scores: BLEU {total_bleu_score/i}, Sacre BLEU SCORE {total_sacred_score/ i}")
    print(f"Current index: {i}")
    time.sleep(10)



print(f"Average BLEU SCORE: {total_bleu_score/ len(sentences)}")
print(f"Average sacre BLEU SCORE: {total_sacred_score / len(sentences)}")
