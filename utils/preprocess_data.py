import argparse
import os
from difflib import SequenceMatcher

import Levenshtein
import numpy as np
from tqdm import tqdm

from helpers import write_lines, read_parallel_lines, encode_verb_form, \
    apply_reverse_transformation, SEQ_DELIMETERS, START_TOKEN


def perfect_align(t, T, insertions_allowed=0,
                  cost_function=Levenshtein.distance):
    # dp[i, j, k] is a minimal cost of matching first `i` tokens of `t` with
    # first `j` tokens of `T`, after making `k` insertions after last match of
    # token from `t`. In other words t[:i] aligned with T[:j].

    # Initialize with INFINITY (unknown)
    shape = (len(t) + 1, len(T) + 1, insertions_allowed + 1)
    dp = np.ones(shape, dtype=int) * int(1e9)
    come_from = np.ones(shape, dtype=int) * int(1e9)
    come_from_ins = np.ones(shape, dtype=int) * int(1e9)

    dp[0, 0, 0] = 0  # The only known starting point. Nothing matched to nothing.
    for i in range(len(t) + 1):  # Go inclusive
        for j in range(len(T) + 1):  # Go inclusive
            for q in range(insertions_allowed + 1):  # Go inclusive
                if i < len(t):
                    # Given matched sequence of t[:i] and T[:j], match token
                    # t[i] with following tokens T[j:k].
                    for k in range(j, len(T) + 1):
                        transform = \
                            apply_transformation(t[i], '   '.join(T[j:k]))
                        if transform:
                            cost = 0
                        else:
                            cost = cost_function(t[i], '   '.join(T[j:k]))
                        current = dp[i, j, q] + cost
                        if dp[i + 1, k, 0] > current:
                            dp[i + 1, k, 0] = current
                            come_from[i + 1, k, 0] = j
                            come_from_ins[i + 1, k, 0] = q
                if q < insertions_allowed:
                    # Given matched sequence of t[:i] and T[:j], create
                    # insertion with following tokens T[j:k].
                    for k in range(j, len(T) + 1):
                        cost = len('   '.join(T[j:k]))
                        current = dp[i, j, q] + cost
                        if dp[i, k, q + 1] > current:
                            dp[i, k, q + 1] = current
                            come_from[i, k, q + 1] = j
                            come_from_ins[i, k, q + 1] = q

    # Solution is in the dp[len(t), len(T), *]. Backtracking from there.
    alignment = []
    i = len(t)
    j = len(T)
    q = dp[i, j, :].argmin()
    while i > 0 or q > 0:
        is_insert = (come_from_ins[i, j, q] != q) and (q != 0)
        j, k, q = come_from[i, j, q], j, come_from_ins[i, j, q]
        if not is_insert:
            i -= 1

        if is_insert:
            alignment.append(['INSERT', T[j:k], (i, i)])
        else:
            alignment.append([f'REPLACE_{t[i]}', T[j:k], (i, i + 1)])

    assert j == 0

    return dp[len(t), len(T)].min(), list(reversed(alignment))


def _split(token):
    if not token:
        return []
    parts = token.split()
    return parts or [token]


def apply_merge_transformation(source_tokens, target_words, shift_idx):
    edits = []
    if len(source_tokens) > 1 and len(target_words) == 1:
        # check merge
        transform = check_merge(source_tokens, target_words)
        if transform:
            for i in range(len(source_tokens) - 1):
                edits.append([(shift_idx + i, shift_idx + i + 1), transform])
            return edits

    if len(source_tokens) == len(target_words) == 2:
        # check swap
        transform = check_swap(source_tokens, target_words)
        if transform:
            edits.append([(shift_idx, shift_idx + 1), transform])
    return edits


def is_sent_ok(sent, delimeters=SEQ_DELIMETERS):
    for del_val in delimeters.values():
        if del_val in sent and del_val != " ":
            return False
    return True


def check_casetype(source_token, target_token):
    if source_token.lower() != target_token.lower():
        return None
    if source_token.lower() == target_token:
        return "$TRANSFORM_CASE_LOWER"
    elif source_token.capitalize() == target_token:
        return "$TRANSFORM_CASE_CAPITAL"
    elif source_token.upper() == target_token:
        return "$TRANSFORM_CASE_UPPER"
    elif source_token[1:].capitalize() == target_token[1:] and source_token[0] == target_token[0]:
        return "$TRANSFORM_CASE_CAPITAL_1"
    elif source_token[:-1].upper() == target_token[:-1] and source_token[-1] == target_token[-1]:
        return "$TRANSFORM_CASE_UPPER_-1"
    else:
        return None


def check_equal(source_token, target_token):
    if source_token == target_token:
        return "$KEEP"
    else:
        return None


def check_split(source_token, target_tokens):
    if source_token.split("-") == target_tokens:
        return "$TRANSFORM_SPLIT_HYPHEN"
    else:
        return None


def check_merge(source_tokens, target_tokens):
    if "".join(source_tokens) == "".join(target_tokens):
        return "$MERGE_SPACE"
    elif "-".join(source_tokens) == "-".join(target_tokens):
        return "$MERGE_HYPHEN"
    else:
        return None


def check_swap(source_tokens, target_tokens):
    if source_tokens == [x for x in reversed(target_tokens)]:
        return "$MERGE_SWAP"
    else:
        return None


def check_plural(source_token, target_token):
    if source_token.endswith("s") and source_token[:-1] == target_token:
        return "$TRANSFORM_AGREEMENT_SINGULAR"
    elif target_token.endswith("s") and source_token == target_token[:-1]:
        return "$TRANSFORM_AGREEMENT_PLURAL"
    else:
        return None


def check_verb(source_token, target_token):
    encoding = encode_verb_form(source_token, target_token)
    if encoding:
        return f"$TRANSFORM_VERB_{encoding}"
    else:
        return None


def apply_transformation(source_token, target_token):
    target_tokens = target_token.split()
    if len(target_tokens) > 1:
        # check split
        transform = check_split(source_token, target_tokens)
        if transform:
            return transform
    checks = [check_equal, check_casetype, check_verb, check_plural]
    for check in checks:
        transform = check(source_token, target_token)
        if transform:
            return transform
    return None


def align_sequences(source_sent, target_sent):
    # check if sent is OK
    if not is_sent_ok(source_sent) or not is_sent_ok(target_sent):
        return None
    source_tokens = source_sent.split()
    target_tokens = target_sent.split()
    matcher = SequenceMatcher(None, source_tokens, target_tokens)
    diffs = list(matcher.get_opcodes())
    all_edits = []
    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        source_part = _split(" ".join(source_tokens[i1:i2]))
        target_part = _split(" ".join(target_tokens[j1:j2]))
        if tag == 'equal':
            continue
        elif tag == 'delete':
            # delete all words separatly
            for j in range(i2 - i1):
                edit = [(i1 + j, i1 + j + 1), '$DELETE']
                all_edits.append(edit)
        elif tag == 'insert':
            # append to the previous word
            for target_token in target_part:
                edit = ((i1 - 1, i1), f"$APPEND_{target_token}")
                all_edits.append(edit)
        else:
            # check merge first of all
            edits = apply_merge_transformation(source_part, target_part,
                                               shift_idx=i1)
            if edits:
                all_edits.extend(edits)
                continue

            # normalize alignments if need (make them singleton)
            _, alignments = perfect_align(source_part, target_part,
                                          insertions_allowed=0)
            for alignment in alignments:
                new_shift = alignment[2][0]
                edits = convert_alignments_into_edits(alignment,
                                                      shift_idx=i1 + new_shift)
                all_edits.extend(edits)

    # get labels
    labels = convert_edits_into_labels(source_tokens, all_edits)
    # match tags to source tokens
    sent_with_tags = add_labels_to_the_tokens(source_tokens, labels)
    return sent_with_tags


def convert_edits_into_labels(source_tokens, all_edits):
    # make sure that edits are flat
    flat_edits = []
    for edit in all_edits:
        (start, end), edit_operations = edit
        if isinstance(edit_operations, list):
            for operation in edit_operations:
                new_edit = [(start, end), operation]
                flat_edits.append(new_edit)
        elif isinstance(edit_operations, str):
            flat_edits.append(edit)
        else:
            raise Exception("Unknown operation type")
    all_edits = flat_edits[:]
    labels = []
    total_labels = len(source_tokens) + 1
    if not all_edits:
        labels = [["$KEEP"] for x in range(total_labels)]
    else:
        for i in range(total_labels):
            edit_operations = [x[1] for x in all_edits if x[0][0] == i - 1
                               and x[0][1] == i]
            if not edit_operations:
                labels.append(["$KEEP"])
            else:
                labels.append(edit_operations)
    return labels


def convert_alignments_into_edits(alignment, shift_idx):
    edits = []
    action, target_tokens, new_idx = alignment
    source_token = action.replace("REPLACE_", "")

    # check if delete
    if not target_tokens:
        edit = [(shift_idx, 1 + shift_idx), "$DELETE"]
        return [edit]

    # check splits
    for i in range(1, len(target_tokens)):
        target_token = " ".join(target_tokens[:i + 1])
        transform = apply_transformation(source_token, target_token)
        if transform:
            edit = [(shift_idx, shift_idx + 1), transform]
            edits.append(edit)
            target_tokens = target_tokens[i + 1:]
            for target in target_tokens:
                edits.append([(shift_idx, shift_idx + 1), f"$APPEND_{target}"])
            return edits

    transform_costs = []
    transforms = []
    for target_token in target_tokens:
        transform = apply_transformation(source_token, target_token)
        if transform:
            cost = 0
            transforms.append(transform)
        else:
            cost = Levenshtein.distance(source_token, target_token)
            transforms.append(None)
        transform_costs.append(cost)
    min_cost_idx = transform_costs.index(min(transform_costs))
    # append to the previous word
    for i in range(0, min_cost_idx):
        target = target_tokens[i]
        edit = [(shift_idx - 1, shift_idx), f"$APPEND_{target}"]
        edits.append(edit)
    # replace/transform target word
    transform = transforms[min_cost_idx]
    target = transform if transform is not None \
        else f"$REPLACE_{target_tokens[min_cost_idx]}"
    edit = [(shift_idx, 1 + shift_idx), target]
    edits.append(edit)
    # append to this word
    for i in range(min_cost_idx + 1, len(target_tokens)):
        target = target_tokens[i]
        edit = [(shift_idx, 1 + shift_idx), f"$APPEND_{target}"]
        edits.append(edit)
    return edits


def add_labels_to_the_tokens(source_tokens, labels, delimeters=SEQ_DELIMETERS):
    tokens_with_all_tags = []
    source_tokens_with_start = [START_TOKEN] + source_tokens
    for token, label_list in zip(source_tokens_with_start, labels):
        all_tags = delimeters['operations'].join(label_list)
        comb_record = token + delimeters['labels'] + all_tags
        tokens_with_all_tags.append(comb_record)
    return delimeters['tokens'].join(tokens_with_all_tags)


def convert_data_from_raw_files(source_file, target_file, output_file, chunk_size):
    tagged = []
    source_data, target_data = read_parallel_lines(source_file, target_file)
    print(f"The size of raw dataset is {len(source_data)}")
    cnt_total, cnt_all, cnt_tp = 0, 0, 0
    for source_sent, target_sent in tqdm(zip(source_data, target_data)):
        try:
            aligned_sent = align_sequences(source_sent, target_sent)
        except Exception:
            aligned_sent = align_sequences(source_sent, target_sent)
        if source_sent != target_sent:
            cnt_tp += 1
        alignments = [aligned_sent]
        cnt_all += len(alignments)
        try:
            check_sent = convert_tagged_line(aligned_sent)
        except Exception:
            # debug mode
            aligned_sent = align_sequences(source_sent, target_sent)
            check_sent = convert_tagged_line(aligned_sent)

        if "".join(check_sent.split()) != "".join(
                target_sent.split()):
            # do it again for debugging
            aligned_sent = align_sequences(source_sent, target_sent)
            check_sent = convert_tagged_line(aligned_sent)
            print(f"Incorrect pair: \n{target_sent}\n{check_sent}")
            continue
        if alignments:
            cnt_total += len(alignments)
            tagged.extend(alignments)
        if len(tagged) > chunk_size:
            write_lines(output_file, tagged, mode='a')
            tagged = []

    print(f"Overall extracted {cnt_total}. "
          f"Original TP {cnt_tp}."
          f" Original TN {cnt_all - cnt_tp}")
    if tagged:
        write_lines(output_file, tagged, 'a')


def convert_labels_into_edits(labels):
    all_edits = []
    for i, label_list in enumerate(labels):
        if label_list == ["$KEEP"]:
            continue
        else:
            edit = [(i - 1, i), label_list]
            all_edits.append(edit)
    return all_edits


def get_target_sent_by_levels(source_tokens, labels):
    relevant_edits = convert_labels_into_edits(labels)
    target_tokens = source_tokens[:]
    leveled_target_tokens = {}
    if not relevant_edits:
        target_sentence = " ".join(target_tokens)
        return leveled_target_tokens, target_sentence
    max_level = max([len(x[1]) for x in relevant_edits])
    for level in range(max_level):
        rest_edits = []
        shift_idx = 0
        for edits in relevant_edits:
            (start, end), label_list = edits
            label = label_list[0]
            target_pos = start + shift_idx
            source_token = target_tokens[target_pos] if target_pos >= 0 else START_TOKEN
            if label == "$DELETE":
                del target_tokens[target_pos]
                shift_idx -= 1
            elif label.startswith("$APPEND_"):
                word = label.replace("$APPEND_", "")
                target_tokens[target_pos + 1: target_pos + 1] = [word]
                shift_idx += 1
            elif label.startswith("$REPLACE_"):
                word = label.replace("$REPLACE_", "")
                target_tokens[target_pos] = word
            elif label.startswith("$TRANSFORM"):
                word = apply_reverse_transformation(source_token, label)
                if word is None:
                    word = source_token
                target_tokens[target_pos] = word
            elif label.startswith("$MERGE_"):
                # apply merge only on last stage
                if level == (max_level - 1):
                    target_tokens[target_pos + 1: target_pos + 1] = [label]
                    shift_idx += 1
                else:
                    rest_edit = [(start + shift_idx, end + shift_idx), [label]]
                    rest_edits.append(rest_edit)
            rest_labels = label_list[1:]
            if rest_labels:
                rest_edit = [(start + shift_idx, end + shift_idx), rest_labels]
                rest_edits.append(rest_edit)

        leveled_tokens = target_tokens[:]
        # update next step
        relevant_edits = rest_edits[:]
        if level == (max_level - 1):
            leveled_tokens = replace_merge_transforms(leveled_tokens)
        leveled_labels = convert_edits_into_labels(leveled_tokens,
                                                   relevant_edits)
        leveled_target_tokens[level + 1] = {"tokens": leveled_tokens,
                                            "labels": leveled_labels}

    target_sentence = " ".join(leveled_target_tokens[max_level]["tokens"])
    return leveled_target_tokens, target_sentence


def replace_merge_transforms(tokens):
    if all(not x.startswith("$MERGE_") for x in tokens):
        return tokens
    target_tokens = tokens[:]
    allowed_range = (1, len(tokens) - 1)
    for i in range(len(tokens)):
        target_token = tokens[i]
        if target_token.startswith("$MERGE"):
            if target_token.startswith("$MERGE_SWAP") and i in allowed_range:
                target_tokens[i - 1] = tokens[i + 1]
                target_tokens[i + 1] = tokens[i - 1]
                target_tokens[i: i + 1] = []
    target_line = " ".join(target_tokens)
    target_line = target_line.replace(" $MERGE_HYPHEN ", "-")
    target_line = target_line.replace(" $MERGE_SPACE ", "")
    return target_line.split()


def convert_tagged_line(line, delimeters=SEQ_DELIMETERS):
    label_del = delimeters['labels']
    source_tokens = [x.split(label_del)[0]
                     for x in line.split(delimeters['tokens'])][1:]
    labels = [x.split(label_del)[1].split(delimeters['operations'])
              for x in line.split(delimeters['tokens'])]
    assert len(source_tokens) + 1 == len(labels)
    levels_dict, target_line = get_target_sent_by_levels(source_tokens, labels)
    return target_line


def main(args):
    convert_data_from_raw_files(args.source, args.target, args.output_file, args.chunk_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source',
                        help='Path to the source file',
                        required=True)
    parser.add_argument('-t', '--target',
                        help='Path to the target file',
                        required=True)
    parser.add_argument('-o', '--output_file',
                        help='Path to the output file',
                        required=True)
    parser.add_argument('--chunk_size',
                        type=int,
                        help='Dump each chunk size.',
                        default=1000000)
    args = parser.parse_args()
    main(args)
