# %% [markdown]
#  Metin Madenciligi Dersi
#
#  Nefret Soylemi ve Ofansif Dil Problemi
#  - Oguzhan KANDAKOGLU
#  - Elif Yildirim
#
#  Bu calisma dort farkli yontem uzerinde denenmistir.
#  - Logistic Regression
#  - Linear SVC (Support Vector Classification)
#  - Naive Bayes
#  - Random Forest
#
#
#   Son kisimda uc farkli siniflandirma algoritmasinin (classifier) da performansi gosterilmektedir.

# %% [markdown]
# # Gerekli Araclarin Dahil Edilmesi

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

import nltk
from nltk.stem.porter import *

import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS

from textstat.textstat import *

import matplotlib.pyplot as plt
import seaborn

%matplotlib inline


# %% [markdown]
# # Veri setinin yuklenmesi

# %%
with open("data/labeled_data.p", 'rb') as f:
    df = pd.read_pickle(f)

# %%
df

# %%
df.describe()


# %%
df.columns


# %% [markdown]
# # Kolon Yapisi
#
# count = Her tweetin kac adet CrowdFlower kullanicisi tarafindan etiketlendiginin sayisi. (Her tweet en az 3 adet kullanici tarafindan etiketlenmistir.)
#
# hate_speech = Kac CF kullanicisinin tweeti 'Nefret Soylemi' olarak etiketlediginin sayisi.
#
# offensive_language = Kac CF kullanicisinin tweeti 'Ofansif Dil' olarak etiketlediginin sayisi.
#
# neither = Kac CF kullanicisinin tweeti 'Ne Nefret Soylemi ne de Ofansif Dil' olarak etiketlediginin sayisi.
#
# # Siniflandirma Yapisi
#
#     0 - Nefret Soylemi
#     1 - Ofansif Dil
#     2 - Neither
#
#

# %%
df['class'].hist()

# %% [markdown]
# # Metin Koleksiyonunun Belirlenmesi
#
# Histogram uzerinden de goruldugu gibi 'Ofansif Dil' kullanilmis tweetlerin sayisi cok daha fazla. 'Ne Ofansif Dil ne de Nefret Soylemi' barindiran tweetler ise 'Nefret Soylemi' barindiran tweetlerden daha fazla.
#
# Bu durum 'Metin Madenciligi' icin cok da olumlu bir durum degil. Ancak buradan soyle bir sonuc cikarabiliriz.
# Ancak aciklamaya baslamadan once bir adim oncesine gitmemiz gerekiyor. Yani:
#
# Verilerin Toplanmasi
# Metin Madenciligi yaparken dikkat etmemiz gereken hususlardan birisi olusturacagimiz modelin hangi alanda calisacagidir. Ornek vermek gerekirse 'TIP' dunyasinda kullanilan terimlerle 'BILISIM' dunyasinda kullanilan kelimeler yazilis olarak ayni olsa da anlam olarak farkli seyleri ifade edebilirler. Dolayisiyle toplayacagimiz verileri ilgili alandan, ilgili konudan toplamamiz gerekir.
#
# Twitter ise her insanin her konudan bahsedebildigi bir platformdur. Icerik olarak cok zengindir. Ancak filtreleme yapmak biraz zordur.
#
# Veri setini olusturan kisi bu konu su sekilde bir yol bulmus:
#
# Tweetler https://hatebase.org adresinde bulunan kelimeleri icerip icermeme durumlarina gore twitter uzerinde 'Scrape' edilmistir.
#
# ---------------------------------
#
# Insan dilinin ne kadar karmasik bir yapi oldugunu sadece yukaridaki histograma bakarak da anlayabiliriz. 'Tweetler nefret soylemi/kufur barindiran tweetler arasindan seciliyor ve ona ragmen "Ne ofansif ne nefret soylemi" barindiran tweetler sadece nefret soylemi barindiran tweetlerden daha fazla cikiyor. 😊 🌪⛈'
#
# ---------------------------------
#
# Tum bunlara ragmen cikan sonuclar bizim gibi baslangic seviyesindeki iki ogrenci icin tatmin edici durumda.
#
#

# %%
tweets = df.tweet

# %% [markdown]
# # Metin On Isleme

# %%
# Stop wordsler belirlenir
stopwords = nltk.corpus.stopwords.words('english')

# Twitter'a ozel terimler belirlenir.
exclusions_for_twitter = ['#ff', 'ff', 'rt']

# Belirlenen terimler stopwords listesine eklenir.
stopwords.extend(exclusions_for_twitter)

stemmer = PorterStemmer()


def pre_process(txt_string):
    """
    RegEXP
        - URL
        - Strip
        - Mentions

        !hastaghler nefret soylemi barindirabileceginden dolayi hashtagler dahil edilmemistir.
    """

    space_pattern = '\s+'
    url_pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]| [!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    mention_pattern = '@[\w\-]+'

    # birden fazla bosluk varsa bunu teke dusur
    parsed_text = re.sub(space_pattern, ' ', txt_string)
    parsed_text = re.sub(url_pattern, '', parsed_text)  # URLleri sil
    parsed_text = re.sub(mention_pattern, '', parsed_text)  # Mentionleri sil

    return parsed_text


def tokenize(tweet):
    # noktalama isaretleri, bosluklar silinir, kelimeler kokune indirilir (stemming) ve tweetin tamami kucuk harfe cekilir.
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens


def basic_tokenize(tweet):
    # Stemming yapmadan tokenlestirme # etiketler icin kullan
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()


# %% [markdown]
# # Terim Dokuman Matrisinin Olusturulmasi

# %%
vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    preprocessor=pre_process,
    ngram_range=(1, 3),  # unigrams, bigrams, trigramslaolusturulur.
    stop_words=stopwords,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=10000,
    min_df=5,
    max_df=0.75
)

# Terim dokuman matrisi
tfidf = vectorizer.fit_transform(tweets).toarray()
# featurelarla sozluk olusuturuluyor
vocab = {v: i for i, v in enumerate(vectorizer.get_feature_names_out())}
idf_vals = vectorizer.idf_

# Terim dokuman puanlari
# idf_dict = {i: idf_vals[i] for i in vocab.values()}


# %% [markdown]
# # Dil Bilgisel Etiketleme

# %%
tweet_tags = []  # tweetlerin dil bilgisel etiketleri
for t in tweets:
    tokens = basic_tokenize(pre_process(t))
    tags = nltk.pos_tag(tokens)
    tag_list = [x[1] for x in tags]
    tag_str = " ".join(tag_list)
    tweet_tags.append(tag_str)

pos_vectorizer = TfidfVectorizer(
    tokenizer=None,
    lowercase=False,
    preprocessor=None,
    ngram_range=(1, 3),
    stop_words=None,
    use_idf=False,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=5000,
    min_df=5,
    max_df=0.75,
)

pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()

pos_vocab = {v: i for i, v in enumerate(
    pos_vectorizer.get_feature_names_out())}


# %% [markdown]
# # Ozellik Cikarimi

# %%
sentiment_analyzer = VS()


def count_twitter_objs(text_string):
    """
    Bu kisimda twittera ait ozelliklerin sayisi bulunur.
    1) urller URLHERE
    2) mentionlar MENTIONHERE
    3) hashtagler HASHTAGHERE
        ile degistirilir.

    Boylelikle tweet icersinde bu ozelliklerden kacar adet oldugu hesaplanir.
    Hesaplanan degerler tuple olarak return edilir.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return(parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE'))


def other_features(tweet):
    """
        Tweete ait ekstra ozellikleri bu kisimda Hesaplariz.
        1) Tweet'in duygu analizi sonuclari: VADER (Valence Aware Dictionary and sEntiment Reasoner)
            - Pos
            - Neg
            - Neut
            - Compound: pozitif, negatif, ve neutrik puanlari toplaminda ortalama puan -1 ile 1 arasinda deger alir.
        2) Kelime sayisi
        3) Hece sayisi
        3) Harf/Karakter sayisi
        4) Tweetin uzunlugu
        5) Ortalama Hece sayisi
        6) Essiz terim sayisi
        7) Terim sayisi
        8) Okunabilirlik puanlari
    """
    sentiment = sentiment_analyzer.polarity_scores(tweet)

    words = pre_process(tweet)  # Get text only

    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001), 4)
    num_unique_terms = len(set(words.split()))

    # Okunabilirlik puanlari
    # https://readable.com/readability/flesch-reading-ease-flesch-kincaid-grade-level/
    # FKRA: Okuma Seviyesi
    # FRE: Kolaylik Seviyesi
    FKRA = round(float(0.39 * float(num_words)/1.0) +
                 float(11.8 * avg_syl) - 15.59, 1)
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) -
                (84.6*float(avg_syl)), 2)

    twitter_objs = count_twitter_objs(tweet)
    retweet = 0
    if "rt" in words:
        retweet = 1

    features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]

    return features

# tweetlere ait ozellikleri cikaririz/hesaplariz.


def get_feature_array(tweets):
    feats = []
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)


# olusturdugumuz ozelliklerin isimleri
other_features_names = ["FKRA", "FRE", "num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total",
                        "num_terms", "num_words", "num_unique_words", "vader neg", "vader pos", "vader neu",
                        "vader compound", "num_hashtags", "num_mentions", "num_urls", "is_retweet"]

# elde edilen tum featurelar
feats = get_feature_array(tweets)


M = np.concatenate([tfidf, pos, feats], axis=1)


# %%
M.shape


# %%
# Tum ozellik isimlerinin bir listesini olusturuyoruz.
variables = ['']*len(vocab)
for k, v in vocab.items():
    variables[v] = k

pos_variables = ['']*len(pos_vocab)
for k, v in pos_vocab.items():
    pos_variables[v] = k

feature_names = variables+pos_variables+other_features_names


# %% [markdown]
# # Modelin Calisitirilmasi

# %%
X = pd.DataFrame(M)
y = df['class'].astype(int)


# %%
# Bu kisimda feature onemlerini Logistic Regression ile hesapliyoruz.
# l1: katsayıların büyüklüğünün mutlak değerine eşit
select = SelectFromModel(LogisticRegression(
    class_weight='balanced', penalty="l1", C=0.01, solver='liblinear'))
X_ = select.fit_transform(X, y)


# %% [markdown]
# # Linear SVC (Support Vector Classifier)

# %%
# l2: katsayıların büyüklüğünün karesine eşit
model = LinearSVC(class_weight='balanced', C=0.01,
                  penalty='l2', max_iter=3000).fit(X_, y)


# %%
y_preds = model.predict(X_)


# %%
report = classification_report(y, y_preds)
print(report)


# %%
confusion_matrix = confusion_matrix(y, y_preds)
matrix_proportions = np.zeros((3, 3))
for i in range(0, 3):
    matrix_proportions[i, :] = confusion_matrix[i, :] / \
        float(confusion_matrix[i, :].sum())
# NeNo(Neutral): Ne Nefret Ne Ofansif
names = ['Nefret', 'Ofansif', 'NeNo']
confusion_df = pd.DataFrame(matrix_proportions, index=names, columns=names)
plt.figure(figsize=(5, 5))
seaborn.heatmap(confusion_df, annot=True, annot_kws={
                "size": 12}, cmap='gist_gray_r', cbar=False, square=True, fmt='.2f')
plt.ylabel(r'Dogru Kategoriler', fontsize=14)
plt.xlabel(r'Tahmin Edilen Kategoriler', fontsize=14)
plt.tick_params(labelsize=12)


# %% [markdown]
# # Logistic Regression

# %%
model = LogisticRegression(class_weight='balanced',
                           penalty='l2', C=0.01, max_iter=3000).fit(X_, y)


# %%
y_preds = model.predict(X_)


# %%
report = classification_report(y, y_preds)
print(report)


# %%
confusion_matrix = confusion_matrix(y, y_preds)
matrix_proportions = np.zeros((3, 3))
for i in range(0, 3):
    matrix_proportions[i, :] = confusion_matrix[i, :] / \
        float(confusion_matrix[i, :].sum())
# NeNo(Neutral): Ne Nefret Ne Ofansif
names = ['Nefret', 'Ofansif', 'NeNo']
confusion_df = pd.DataFrame(matrix_proportions, index=names, columns=names)
plt.figure(figsize=(5, 5))
seaborn.heatmap(confusion_df, annot=True, annot_kws={
                "size": 12}, cmap='gist_gray_r', cbar=False, square=True, fmt='.2f')
plt.ylabel(r'Dogru Kategoriler', fontsize=14)
plt.xlabel(r'Tahmin Edilen Kategoriler', fontsize=14)
plt.tick_params(labelsize=12)


# %% [markdown]
# # Naive Bayes

# %%
# import naive bayes

# %%
model = BernoulliNB().fit(X_, y)
y_preds = model.predict(X_)
report = classification_report(y, y_preds)
print(report)


# %%
confusion_matrix = confusion_matrix(y, y_preds)
matrix_proportions = np.zeros((3, 3))
for i in range(0, 3):
    matrix_proportions[i, :] = confusion_matrix[i, :] / \
        float(confusion_matrix[i, :].sum())
# NeNo(Neutral): Ne Nefret Ne Ofansif
names = ['Nefret', 'Ofansif', 'NeNo']
confusion_df = pd.DataFrame(matrix_proportions, index=names, columns=names)
plt.figure(figsize=(5, 5))
seaborn.heatmap(confusion_df, annot=True, annot_kws={
                "size": 12}, cmap='gist_gray_r', cbar=False, square=True, fmt='.2f')
plt.ylabel(r'Dogru Kategoriler', fontsize=14)
plt.xlabel(r'Tahmin Edilen Kategoriler', fontsize=14)
plt.tick_params(labelsize=12)


# %% [markdown]
# # Ensemble Learning / Random Forest

# %%
# ENSEMBLE LEARNING
# import ensemble learning

# %%
model = RandomForestClassifier().fit(X_, y)
y_preds = model.predict(X_)
report = classification_report(y, y_preds)
print(report)


# %%
confusion_matrix = confusion_matrix(y, y_preds)
matrix_proportions = np.zeros((3, 3))
for i in range(0, 3):
    matrix_proportions[i, :] = confusion_matrix[i, :] / \
        float(confusion_matrix[i, :].sum())
# NeNo(Neutral): Ne Nefret Ne Ofansif
names = ['Nefret', 'Ofansif', 'NeNo']
confusion_df = pd.DataFrame(matrix_proportions, index=names, columns=names)
plt.figure(figsize=(5, 5))
seaborn.heatmap(confusion_df, annot=True, annot_kws={
                "size": 12}, cmap='gist_gray_r', cbar=False, square=True, fmt='.2f')
plt.ylabel(r'Dogru Kategoriler', fontsize=14)
plt.xlabel(r'Tahmin Edilen Kategoriler', fontsize=14)
plt.tick_params(labelsize=12)


# %% [markdown]
# # Degerlendirme ve Yorumlama
#
# - Random Forest Classifier %100 oranda accuracy gosterdi. Ancak veri setindeki verilerin kismi tekrarinin boyle bir sorun olusturabilecegini dusunuyoruz.
# - Hem SVC hem de Linear SVC denedik. SVC: %80, LSVC %86 ACC. gosterdi.
# - Naive Bayes: %77 acc. gosterdi. Featurelarimiz negatatif degerler iceriyor. Bunlar re-map edilip MultinominalNB / CategoricalNB de denenebilir.
# - Logistic Regression ise %78 acc.
#
# Tum bu puanlarin veri setindeki sinif sayilari arasindaki tutarsizlik olabilecegini dusunuyoruz.
