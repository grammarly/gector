#!/usr/bin/env python
"""
Convert CLC-FCE dataset (The Cambridge Learner Corpus) to the parallel sentences format.
"""

import argparse
import glob
import os
import re
from xml.etree import cElementTree

from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm


def annotate_fce_doc(xml):
    """Takes a FCE xml document and yields sentences with annotated errors."""
    result = []
    doc = cElementTree.fromstring(xml)
    paragraphs = doc.findall('head/text/*/coded_answer/p')
    for p in paragraphs:
        text = _get_formatted_text(p)
        result.append(text)

    return '\n'.join(result)


def _get_formatted_text(elem, ignore_tags=None):
    text = elem.text or ''
    ignore_tags = [tag.upper() for tag in (ignore_tags or [])]
    correct = None
    mistake = None

    for child in elem.getchildren():
        tag = child.tag.upper()
        if tag == 'NS':
            text += _get_formatted_text(child)

        elif tag == 'UNKNOWN':
            text += ' UNKNOWN '

        elif tag == 'C':
            assert correct is None
            correct = _get_formatted_text(child)

        elif tag == 'I':
            assert mistake is None
            mistake = _get_formatted_text(child)

        elif tag in ignore_tags:
            pass

        else:
            raise ValueError(f"Unknown tag `{child.tag}`", text)

    if correct or mistake:
        correct = correct or ''
        mistake = mistake or ''
        if '=>' not in mistake:
            text += f'{{{mistake}=>{correct}}}'
        else:
            text += mistake

    text += elem.tail or ''
    return text


def convert_fce(fce_dir):
    """Processes the whole FCE directory. Yields annotated documents (strings)."""

    # Ensure we got the valid dataset path
    if not os.path.isdir(fce_dir):
        raise UserWarning(
            f"{fce_dir} is not a valid path")

    dataset_dir = os.path.join(fce_dir, 'dataset')
    if not os.path.exists(dataset_dir):
        raise UserWarning(
            f"{fce_dir} doesn't point to a dataset's root dir")

    # Convert XML docs to the corpora format
    filenames = sorted(glob.glob(os.path.join(dataset_dir, '*/*.xml')))

    docs = []
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            doc = annotate_fce_doc(f.read())
            docs.append(doc)
    return docs


def main():
    fce = convert_fce(args.fce_dataset_path)
    with open(args.output + "/fce-original.txt", 'w', encoding='utf-8') as out_original, \
            open(args.output + "/fce-applied.txt", 'w', encoding='utf-8') as out_applied:
        for doc in tqdm(fce, unit='doc'):
            sents = re.split(r"\n +\n", doc)
            for sent in sents:
                tokenized_sents = sent_tokenize(sent)
                for i in range(len(tokenized_sents)):
                    if re.search(r"[{>][.?!]$", tokenized_sents[i]):
                        tokenized_sents[i + 1] = tokenized_sents[i] + " " + tokenized_sents[i + 1]
                        tokenized_sents[i] = ""
                    regexp = r'{([^{}]*?)=>([^{}]*?)}'
                    original = re.sub(regexp, r"\1", tokenized_sents[i])
                    applied = re.sub(regexp, r"\2", tokenized_sents[i])
                    # filter out nested alerts
                    if original != "" and applied != "" and not re.search(r"[{}=]", original) \
                            and not re.search(r"[{}=]", applied):
                        out_original.write(" ".join(word_tokenize(original)) + "\n")
                        out_applied.write(" ".join(word_tokenize(applied)) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        "Convert CLC-FCE dataset to the parallel sentences format."))
    parser.add_argument('fce_dataset_path',
                        help='Path to the folder with the FCE dataset')
    parser.add_argument('--output',
                        help='Path to the output folder')
    args = parser.parse_args()

    main()
