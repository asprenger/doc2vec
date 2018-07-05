
import re
import argparse
import itertools
import spacy
 
def readline_gen(input_path):
    for line in open(input_path, encoding="utf-8"):
        line = line.strip()
        yield line

def cleanup_text(text):
    '''Cleanup text before tokenization'''
    # Remove HTML tags
    norm_text = re.sub(r'<[^>]*>', ' ', text)
    # Remove links
    norm_text = re.sub('http[s]?://\S*', ' ', norm_text)
    # Replace HTML character encoding
    for s in ['&quot;', '"', '&amp;', '&', '&nbsp;', ' ', '&lt;', '<', '&gt;', '>']:
        norm_text = norm_text.replace(s, ' ')
    return norm_text

def parse_doc(doc, token_filter_fn, token_output_fn):
    return [token_output_fn(token) for token in doc if token_filter_fn(token)]

def token_filter_fn(token):
    return not token.is_stop \
        and not token.pos_ in ['PUNCT', 'SPACE'] \
        and len(token.text) > 1

def token_output_fn(token):
    return token.lemma_.lower()

def parse_docs(docs, nlp, cores, token_filter_fn, token_output_fn):
    clean_docs = [cleanup_text(doc) for doc in docs]
    docs = nlp.pipe(clean_docs, n_threads=cores)
    tokenized_docs = [parse_doc(doc, token_filter_fn, token_output_fn) for doc in docs]
    return tokenized_docs
    
def run(input_path, output_path, cores, batch_size):
    locale='de'
    newline = '\n'.encode("utf-8")
    nlp = spacy.load(locale, disable=['ner'])
    print('Spacy pipeline: %s' % nlp.pipe_names)
    doc_idx = 0
    gen = readline_gen(input_path)
    with open(output_path, 'wb') as f:
        while True:
            doc_batch = list(itertools.islice(gen, batch_size))
            if len(doc_batch) == 0:
                break
            tokenized_docs = parse_docs(doc_batch, nlp, cores, token_filter_fn, token_output_fn)
            tokenized_docs = [' '.join(tokens) for tokens in tokenized_docs]
            for tokenized_doc in tokenized_docs:
                f.write(tokenized_doc.encode("utf-8"))
                f.write(newline)
                doc_idx += 1
                if doc_idx % 100 == 0:
                    print('%d docs processed' % doc_idx)
    print('%d docs have been written to %s' % (doc_idx, output_path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', help='Number of cores used for parsing', type=int, default=1)
    parser.add_argument('--batch-size', help='Batch size for parsing', type=int, default=1)
    parser.add_argument('input_path', help='Input path')
    parser.add_argument('output_path', help='Output path')
    args = parser.parse_args()
    run(args.input_path, args.output_path, args.cores, args.batch_size)




if __name__ == '__main__':
    main()

