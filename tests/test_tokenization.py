"""Tests for GECToR custom tokenization utilities"""

from gector.tokenization import *
import unittest
from transformers import AutoTokenizer

class TokenizationTests(unittest.TestCase):
    """A test class that tests the GECToR custom tokenization utilities."""

    def test_get_bpe_groups(self):
        """Test get_bpe_groups method with an input sentence."""
        token_offsets = [(0, 6), (7, 15), (16, 18), (19, 21), (22, 23), (24, 29),
        (30, 33), (34, 40), (41, 44), (45, 57), (58, 61), (62, 70), (71, 81),
        (82, 87), (88, 94), (95, 98), (99, 108), (109, 111), (112, 121), (122, 131),
        (132, 133)]
        bpe_offsets = [(0, 6), (7, 15), (16, 18), (19, 21), (22, 23), (24, 29),
        (30, 33), (34, 40), (41, 44), (45, 54), (54, 57), (58, 61), (62, 70),
        (71, 74), (74, 81), (82, 87), (88, 94), (95, 98), (99, 108), (109, 111),
        (112, 121), (122, 131),(132, 133)]
        input_ids = [50265, 37158, 15, 1012, 2156, 89, 32, 460, 5, 12043, 268, 8,
        4131, 22761, 13659, 49, 1351, 8, 11360, 7, 12043, 7768, 479]
        expected_results = ([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9, 10],
        [11], [12], [13, 14], [15], [16], [17], [18], [19], [20], [21], [22]],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22])
        tested_results = get_bpe_groups(token_offsets, bpe_offsets, input_ids)

        self.assertEqual(expected_results == tested_results)

    def test_reduce_input_ids(self):
        """Test reduce_input_ids method with an input sentence."""
        input_ids = [50265, 37158, 15, 1012, 2156, 89, 32, 460, 5, 12043, 268,
        8, 4131, 22761, 13659, 49, 1351, 8, 11360, 7, 12043, 7768, 479]
        bpe_groups = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9, 10], [11],
        [12], [13, 14], [15], [16], [17], [18], [19], [20], [21], [22]]
        saved_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22]
        expected_results = ([50265, 37158, 15, 1012, 2156, 89, 32, 460, 5, 12043,
        268, 8, 4131, 22761, 13659, 49, 1351, 8, 11360, 7, 12043, 7768, 479],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22])
        tested_results = reduce_input_ids(input_ids, bpe_groups, saved_ids)

        self.assertEqual(expected_results == tested_results)

    def test_get_offsets_and_reduce_input_ids(self):
        """Test get_offsets_and_reduce_input_ids method with an input sentence."""
        tokenizer_output = {'input_ids': [[50265, 37158, 15, 1012, 2156, 89, 32,
        460, 5, 12043, 268, 8, 4131, 22761, 13659, 49, 1351, 8, 11360, 7, 12043,
        7768, 479]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 'offset_mapping': [[(0, 6), (7, 15),
        (16, 18), (19, 21), (22, 23), (24, 29), (30, 33), (34, 40), (41, 44),
        (45, 54), (54, 57), (58, 61), (62, 70), (71, 74), (74, 81), (82, 87),
        (88, 94), (95, 98), (99, 108), (109, 111), (112, 121), (122, 131), (132, 133)]]}
        token_offset_list = [[(0, 6), (7, 15), (16, 18), (19, 21), (22, 23),
        (24, 29), (30, 33), (34, 40), (41, 44), (45, 57), (58, 61), (62, 70),
        (71, 81), (82, 87), (88, 94), (95, 98), (99, 108), (109, 111),
        (112, 121), (122, 131), (132, 133)]]
        index_name = "bert"
        max_bpe_length = 512
        max_bpe_pieces = 5
        expected_results = {'bert': [[50265, 37158, 15, 1012, 2156, 89, 32, 460,
        5, 12043, 268, 8, 4131, 22761, 13659, 49, 1351, 8, 11360, 7, 12043, 7768,
        479]], 'bert-offsets': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15,
        16, 17, 18, 19, 20, 21, 22]], 'mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
        tested_results = get_offsets_and_reduce_input_ids(tokenizer_output, token_offset_list,
                                                          index_name=index_name,
                                                          max_bpe_length=max_bpe_length,
                                                          max_bpe_pieces=max_bpe_pieces
                                                          )

        self.assertEqual(assert expected_results == tested_results)

    def test_get_offset_for_tokens(self):
        """Test get_offset_for_tokens method with an input sentence."""
        tokens = ['$START', 'Everyday', 'on', 'TV', ',', 'there', 'are', 'always',
        'the', 'entertainers', 'and', 'athletes','dedicating', 'their', 'effort',
        'and', 'abilities', 'to', 'entertain', 'audiences', '.']
        
        expected_results = [(0, 6), (7, 15), (16, 18), (19, 21), (22, 23), (24, 29),
        (30, 33), (34, 40), (41, 44), (45, 57), (58, 61), (62, 70), (71, 81),
        (82, 87), (88, 94), (95, 98), (99, 108), (109, 111), (112, 121), (122, 131), (132, 133)]
        tested_results = get_offset_for_tokens(tokens)
        self.assertEqual(expected_results == tested_results)

    def test_get_token_offsets(self):
        """Test get_token_offsets method with an input sentence."""
        batch = [['$START', 'Everyday', 'on', 'TV', ',', 'there', 'are', 'always',
        'the', 'entertainers', 'and', 'athletes','dedicating', 'their', 'effort',
        'and', 'abilities', 'to', 'entertain', 'audiences', '.']]
        tested_results = get_token_offsets(batch)
        expected_results = [[(0, 6), (7, 15), (16, 18), (19, 21), (22, 23), (24, 29),
        (30, 33), (34, 40), (41, 44), (45, 57), (58, 61), (62, 70), (71, 81), (82, 87),
        (88, 94), (95, 98), (99, 108), (109, 111), (112, 121), (122, 131), (132, 133)]]

        self.assertEqual(expected_results == tested_results)

    def test_pad_output(self):
        """Test pad_output with an input sentence."""
        output = {'bert': [[50265, 37158, 15, 1012, 2156, 89, 32, 460, 5, 12043,
        268, 8, 4131, 22761, 13659, 49, 1351, 8, 11360, 7, 12043, 7768, 479]],
        'bert-offsets': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17,
        18, 19, 20, 21, 22]],'mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1]]}
        tested_results = pad_output(output)
        expected_results = {'bert': [[50265, 37158, 15, 1012, 2156, 89, 32, 460,
        5, 12043, 268, 8, 4131, 22761, 13659, 49, 1351, 8, 11360, 7, 12043, 7768,
        479]], 'bert-offsets': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16,
        17, 18, 19, 20, 21, 22]], 'mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1]]}

        self.assertEqual(expected_results == tested_results)


    def test_tokenize_batch(self):
        """Test tokenize_batch method with an input sentence."""
        index_name = "bert"
        tokenizer = AutoTokenizer.from_pretrained('roberta-base', do_lower_case=True,
                                                  do_basic_tokenize=False, use_fast=True)
        batch_tokens = [['$START', 'Everyday', 'on', 'TV', ',', 'there', 'are', 'always',
        'the', 'entertainers', 'and', 'athletes','dedicating', 'their', 'effort',
        'and', 'abilities', 'to', 'entertain', 'audiences', '.']]
        max_bpe_length = 512
        max_bpe_pieces = 5
        tested_results = tokenize_batch(tokenizer, batch_tokens, index_name, max_bpe_length, max_bpe_pieces)
        expected_results = {'bert': [[1629, 4014, 11328, 37158, 15, 1012, 2156, 89, 32, 460, 5,
        12043, 268, 8, 4131, 22761, 13659, 49, 1351, 8, 11360, 7, 12043, 7768, 479]],
        'bert-offsets': [[0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23,
        24]], 'mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}

        self.assertEqual(expected_results == tested_results)

        
if __name__ == '__main__':
    unittest.main()
