import spacy

text = """
Dave watched as the forest burned up on the hill,
only a few miles from his house. The car had
been hastily packed and Marta was inside trying to round
up the last of the pets. "Where could she be?" he wondered
as he continued to wait for Marta to appear with the pets.
"""

# step 1: Load data using en_core_web_sm language model which was downloaded
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# step 2: Tokenization
tokens = [token for token in doc] # or you can do just doc.tokens
print(f"Tokens: {tokens}")

# step 3: removing stop words
filtered_tokens = [token for token in tokens if not token.is_stop]
# or filtered_tokens = [token for token in doc.tokens if not token.is_stop]
print(f"Filtered Tokens: {filtered_tokens}")

# step 4: Normalization (with Lemmatization and not Stemming)
lemmas = [f"Token: {token} | lemma: {token.lemma_}" for token in filtered_tokens]
print(f"Lemmas: {lemmas}")

# step 5: Vectorizing text
print(filtered_tokens[1].vector)
