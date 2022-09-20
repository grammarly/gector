import unittest
import torch
from gector.basic_field_embedder import BasicTextFieldEmbedder
from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.tokenizer_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer 
from allennlp.data.vocabulary import Vocabulary
import unittest
from torch import LongTensor
import pudb


class Test_indexer_embedder(unittest.TestCase):
    def test_tokens_to_indices(self):
        pudb.set_trace()
        # initialize for indexer and embedder
        token_embedder = BasicTextFieldEmbedder({'bert': PretrainedBertEmbedder(pretrained_model='roberta-base',

                                                                           requires_grad=False,
                                                                           top_layer_only=True,
                                                                           special_tokens_fix=True)},
                                                embedder_to_indexer_map={"bert": ["bert", "bert-offsets"]}, allow_unmatched_keys=True)

        token_indexer = PretrainedBertIndexer(pretrained_model='roberta-base',
                    do_lowercase=True,
                    max_pieces_per_token=5,
                    special_tokens_fix=True)

        # initialize WhitespaceTokenizer                                      
        tokenizer = WhitespaceTokenizer()
        sentence = "the Quick brown fox jumped over the laziest lazy elmo"
        vocab_path = "../data/output_vocabulary"
        tokens = tokenizer.tokenize(sentence)
        vocab = Vocabulary.from_files(vocab_path)
        model_name = 'roberta-base'
        padding_lengths = {'bert': 12, 'bert-offsets':12, 'mask': 12}


        inputs = token_indexer.tokens_to_indices(tokens, vocab)
        # add paddings so that all tensor have same size
        padded_tokens = token_indexer.as_padded_tensor_dict(inputs,padding_lengths)
        reshaped_padded_tokens = {}
        # reshape padded_tokens to be in the batch format. (since we only pass one sentence in)
        for key, value in padded_tokens.items():
            reshaped_padded_tokens[key] = torch.reshape(value, (1, value.size(0)))

        token_embedder(reshaped_padded_tokens)
if __name__ == "__main__":
    unittest.main()
