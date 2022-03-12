import argparse
import os
import spacy
from difflib import SequenceMatcher
from collections import Counter
from tqdm.auto import tqdm

import numpy as np

nlp = spacy.load("en_core_web_sm", exclude=['tagger', 'parser', 'ner', 'lemmatizer', 'textcat', 'custom'])


def get_tokens(doc):
    all_tokens = []
    for token in doc:
        all_tokens.append(token.text)
        if len(token.whitespace_):
            all_tokens.append(token.whitespace_)
    return all_tokens


def make_changes(nlp, source_sentence, target_sentences=[], min_count=2, debug=False):
    source_tokens = get_tokens(nlp(str(source_sentence)))

    target_docs_tokens = [get_tokens(nlp(str(sent))) for sent in target_sentences]
    all_actions = []

    for i in range(len(target_sentences)):

        target_tokens = target_docs_tokens[i]

        matcher = SequenceMatcher(None, source_tokens, target_tokens)

        raw_diffs = list(matcher.get_opcodes())

        for diff in raw_diffs:
            if diff[0] == 'replace':
                # "source_start_token", "source_end_token", "target_part"
                all_actions.append(
                    ('replace', diff[1], diff[2], "".join(target_tokens[diff[3]: diff[4]]))
                )
            if diff[0] == 'delete':
                # "source_start_token", "source_end_token"
                all_actions.append(
                    ('delete', diff[1], diff[2])
                )
            if diff[0] == 'insert':
                # "source_start_token", "target_part"
                all_actions.append(
                    ('insert', diff[1], "".join(target_tokens[diff[3]: diff[4]]))
                )

    good_actions = [k for k, v in Counter(all_actions).items() if v >= min_count]
    good_actions.sort(key=lambda x: x[1])  # sort by second field - start token

    if debug:
        print("All actions", all_actions)
        print("Good actions", good_actions)

    if len(good_actions) > 0:

        final_text = ""
        current_start = 0
        previous_end = 0

        for action in good_actions:
            current_start = action[1]
            final_text += "".join(source_tokens[previous_end: current_start])
            if action[0] == 'replace':
                final_text += action[3]
                previous_end = action[2]
            if action[0] == 'delete':
                previous_end = action[2]
            if action[0] == 'insert':
                final_text += action[2]
                previous_end = action[1]

        final_text += "".join(source_tokens[previous_end:])
        return final_text

    else:
        return ''.join(source_tokens)


def read_lines(fn):
    if not os.path.exists(fn):
        return []
    with open(fn, 'r', encoding='utf-8') as f:
        text = f.read()
    lines = text.split("\n")
    if lines[-1] == '':
        return lines[:-1]
    else:
        return lines


def write_lines(fn, lines, mode='w'):
    text_to_write = "\n".join(list(lines))
    with open(fn, encoding='utf-8', mode=mode) as f:
        f.write(text_to_write)


def main(args):
    source_sentences = read_lines(args.source_file)

    pred_texts = []
    for target_path in args.target_files:
        pred_texts.append(read_lines(target_path))

    pred_texts = np.array(pred_texts)
    sent_after_merge = []
    for i in tqdm(range(len(source_sentences))):
        source_sentence = source_sentences[i]
        target_sentences = pred_texts[:, i]
        new_sentence = make_changes(nlp, source_sentence, target_sentences=target_sentences, min_count=args.min_count, debug=False)
        sent_after_merge.append(new_sentence)

    output_sentences = [str(t) for t in sent_after_merge]

    write_lines(args.output_file, output_sentences, mode='w')


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file',
                        help='Source file with errors in sentences', 
                        required=True)
    parser.add_argument('--target_files',
                        help='Path to the corrected files by single models', nargs='+',
                        required=True)
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        required=True)
    parser.add_argument('--min_count',
                        type=int,
                        help='Minimal count of same correction by models',
                        default=2)
    args = parser.parse_args()
    main(args)
