import os
from time import time


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def get_bpe_groups(token_offsets, bpe_offsets, input_ids, max_bpe_pieces=5):
    bpe_groups = []
    last_used_bpe = 0
    # find the size of offsets
    if (0, 0) in bpe_offsets:
        bpe_size = bpe_offsets.index((0, 0))
    else:
        bpe_size = len(bpe_offsets)

    saved_ids = [i for i in range(len(input_ids))]
    redundant_ids = []
    for token_offset in token_offsets:
        start_token, end_token = token_offset
        bpe_group = []
        mapping_is_found = False
        for i in range(last_used_bpe, bpe_size):
            start_bpe, end_bpe = bpe_offsets[i]
            if start_bpe >= start_token and end_bpe <= end_token:
                # check if bpe_group is satisfy max_bpe_pieces constraint
                if len(bpe_group) < max_bpe_pieces:
                    bpe_group.append(i)
                else:
                    redundant_ids.append(i)
                last_used_bpe = i + 1
                mapping_is_found = True
            elif mapping_is_found:
                # stop doing useless iterations
                break
            else:
                continue
        bpe_groups.append(bpe_group)
    saved_ids = [i for i in saved_ids if i not in redundant_ids]
    return bpe_groups, saved_ids


def reduce_input_ids(input_ids, bpe_groups, saved_ids,
                     max_bpe_length=80, max_bpe_pieces=5):
    # check if sequence is satisfy max_bpe_length constraint
    while len(saved_ids) > max_bpe_length:
        max_bpe_pieces -= 1
        for token_id in range(len(bpe_groups)):
            if len(bpe_groups[token_id]) > max_bpe_pieces:
                redundant_ids = bpe_groups[token_id][max_bpe_pieces:]
                bpe_groups[token_id] = bpe_groups[token_id][:max_bpe_pieces]
                saved_ids = [i for i in saved_ids if i not in redundant_ids]

    # get offsets
    reduced_ids = [input_ids[i] for i in saved_ids]
    correct_offsets = []
    idx = 0
    for i, bpe_group in enumerate(bpe_groups):
        norm_idx = min(idx, len(reduced_ids) - 1)
        correct_offsets.append(norm_idx)
        idx += len(bpe_group)

    return reduced_ids, correct_offsets


def get_offsets_and_reduce_input_ids(tokenizer_output, token_offset_list,
                                     index_name="bert", max_bpe_length=80,
                                     max_bpe_pieces=5):
    timings = {"bpe": 0, "reduce": 0, "mask": 0}
    output_ids, output_offsets, output_masks = [], [], []
    for i, token_offsets in enumerate(token_offset_list):
        input_ids = tokenizer_output['input_ids'][i]

        t0 = time()
        # get bpe level offsets
        bpe_offsets = tokenizer_output['offset_mapping'][i]
        bpe_groups, saved_ids = get_bpe_groups(token_offsets, bpe_offsets,
                                               input_ids,
                                               max_bpe_pieces=max_bpe_pieces)
        t1 = time()
        timings["bpe"] += t1 - t0

        # reduce sequence length
        reduced_ids, correct_offsets = reduce_input_ids(input_ids, bpe_groups,
                                                        saved_ids,
                                                        max_bpe_length=max_bpe_length,
                                                        max_bpe_pieces=max_bpe_pieces)

        t2 = time()
        timings["reduce"] += t2 - t1

        # get mask
        bpe_mask = [1 for _ in correct_offsets]
        output_ids.append(reduced_ids)
        output_offsets.append(correct_offsets)
        output_masks.append(bpe_mask)

        t3 = time()
        timings["mask"] += t3 - t2

    # tt = sum(timings.values())
    # timings = {k: f"{round(v * 100 / tt, 2)}%" for k, v in timings.items()}
    # print(timings)

    output = {index_name: output_ids,
              f"{index_name}-offsets": output_offsets,
              "mask": output_masks}
    return output


def get_offset_for_tokens(tokens):
    sentence = " ".join(tokens)
    token_offsets = []
    end_idx = 0
    for token in tokens:
        idx = sentence[end_idx:].index(token) + end_idx
        end_idx = idx + len(token)
        offset = (idx, end_idx)
        token_offsets.append(offset)
    return token_offsets


def get_token_offsets(batch):
    token_offset_list = []
    for tokens in batch:
        token_offsets = get_offset_for_tokens(tokens)
        token_offset_list.append(token_offsets)
    return token_offset_list


def pad_output(output, pad_idx=0):
    padded_output = {}
    for input_key in output.keys():
        indexes = output[input_key]
        max_len = max([len(x) for x in indexes])
        padded_indexes = []
        for index_list in indexes:
            cur_len = len(index_list)
            pad_len = max_len - cur_len
            padded_indexes.append(index_list + [pad_idx] * pad_len)
        padded_output[input_key] = padded_indexes
    return padded_output


def tokenize_batch(tokenizer, batch_tokens, index_name="bert",
                   max_bpe_length=80, max_bpe_pieces=5):
    timings = {}
    t0 = time()
    # get batch with sentences
    batch_sentences = [" ".join(x) for x in batch_tokens]
    # get token level offsets
    token_offset_list = get_token_offsets(batch_tokens)
    # token_offset_list = get_token_offsets_multi(batch_tokens)
    t1 = time()
    timings["offset_time"] = t1 - t0
    # tokenize batch
    tokenizer_output = tokenizer.batch_encode_plus(batch_sentences,
                                                   pad_to_max_length=False,
                                                   return_offsets_mapping=True,
                                                   add_special_tokens=False)

    t2 = time()
    timings["tokenize_time"] = t2 - t1
    # postprocess batch
    output = get_offsets_and_reduce_input_ids(tokenizer_output,
                                              token_offset_list,
                                              index_name=index_name,
                                              max_bpe_length=max_bpe_length,
                                              max_bpe_pieces=max_bpe_pieces)

    t3 = time()
    timings["reduce_time"] = t3 - t2
    # pad output
    output = pad_output(output)
    t4 = time()
    timings["pading_time"] = t4 - t3
    # tt = sum(timings.values())
    # timings = {k:f"{round(v*100/tt, 2)}%" for k,v in timings.items()}
    # print(timings)

    return output
