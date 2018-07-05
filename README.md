# doc2vec

This repository contains Python scripts to train a doc2vec models using [Gensim](https://radimrehurek.com/gensim/models/doc2vec.html).

Details about the doc2vec algorithm can be found in the paper [Distributed Representations of Sentences and Documents](https://arxiv.org/abs/1405.4053v2).


# Create a DeWiki dataset

Doc2vec is an unsupervised learning algorithm and a model can be trained with any set of documents. A document can be 
anything from a short 140 character tweet, a single paragraph like an article abstract, a  news article, or a book. 

For German a good baseline is to train a model using the [german Wikipedia](https://de.wikipedia.org/wiki/Wikipedia:Hauptseite).

Download the latest DeWiki dump:

    wget http://download.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles.xml.bz2

Extract the content:

    wget http://medialab.di.unipi.it/Project/SemaWiki/Tools/WikiExtractor.py
    python WikiExtractor.py -c -b 25M -o extracted dewiki-latest-pages-articles.xml.bz2
    find extracted -name '*bz2' \! -exec bzip2 -k -c -d {} \; > dewiki.xml

Clean up XML and extract the text:

    cat dewiki.xml|tr '\n' ' '|sed 's/<\/doc>/<\/doc>\n/g' > dewiki1
    grep -v 'title="Datei:' dewiki1 > dewiki2
    grep -v 'title="Wikipedia:' dewiki2 > dewiki3
    cat dewiki3|sed 's/<[^>]*>//g' > dewiki4
    grep -v "^\s\sKategorie:" dewiki4 > dewiki.txt
    rm dewiki1 dewiki2 dewiki3 dewiki4


# Train a model from the DeWiki dataset

Preprocess the train data:

    python -u preprocess.py dewiki.txt dewiki-preprocessed.txt

Train the model:

    python -u train.py --epochs 40 --vocab-min-count 10 data/stopwords_german.txt dewiki-preprocessed.txt /tmp/models/doc2vec-dewiki
