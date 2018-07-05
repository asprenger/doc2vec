# -*- coding: utf-8 -*-

"""
Prepare a set of documents for training.
"""

import os
import re
import argparse
import logging

def normalize_text(text):
    '''Convert text to lower-case, remove HTML tags and pad punctuation with spaces.'''
    norm_text = text.lower()
    norm_text = re.sub(r'<[^>]*>', ' ', norm_text)
    norm_text = re.sub('http[s]?://\S*', ' ', norm_text)
    for s in ['&quot;', '&amp;', '&nbsp;', '&lt;']:
        norm_text = norm_text.replace(s, ' ')
    for char in ['.', '"', '\'', '/', ',', '(', ')', '[', ']', '{', '}', '!', '?', ';', ':', '«', '»', '„', '“']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')
    return norm_text

def run(input_path, output_path):
    doc_id = 1
    newline = '\n'.encode("utf-8")
    with open(output_path, 'wb') as f:
        for i, line in enumerate(open(input_path, encoding="utf-8")):
            line = line.strip()
            txt_norm = normalize_text(line)
            if len(txt_norm) > 200:
                f.write(txt_norm.encode("utf-8"))
                f.write(newline)
                doc_id += 1
                if doc_id % 10000 == 0:
                    logging.info('%d documents processed', doc_id)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Input path')
    parser.add_argument('output_path', help='Output path')
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s', level=logging.INFO)
    run(args.input_path, args.output_path)

if __name__ == '__main__':
    main()
