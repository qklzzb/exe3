import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Read the Moby Dick text data
moby_dick = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')

# "Punkt, etc., from nltk_data can be manually downloaded and stored in the nltk_data folder under Python."

# 1. Tokenization
words = word_tokenize(moby_dick.lower())  # Tokenize the text and convert it to lowercase

# 2. Stop-words filtering
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

# 3. Parts-of-Speech tagging
pos_tags = nltk.pos_tag(filtered_words)

# 4. POS frequency
fdist = FreqDist(tag for (word, tag) in pos_tags)
top_5_pos = fdist.most_common(5)
print("Top 5 POS Tags and Their Frequencies:")
for pos, freq in top_5_pos:
    print(f"{pos}: {freq}")

# 5. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word, _ in pos_tags]
top_20_words = FreqDist(lemmatized_words).most_common(20)
print("\nTop 20 Lemmatized Words:")
for word, freq in top_20_words:
    print(f"{word}: {freq}")

# 6. Plotting frequency distribution
fdist.plot(30, title="POS Frequency Distribution")
plt.show()
