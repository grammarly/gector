import torch
import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.data import Batch
from allennlp.data.fields import TextField
from allennlp.data import Instance
from allennlp.data import Token
from allennlp.data import Vocabulary

from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.tokenizer_indexer import PretrainedBertIndexer


class TestRobertaEmbedder(ModelTestCase):
    """Test token embedder for RoBERTa model."""

    def setup_method(self):
        """Set up indexer."""

        super().setup_method()

        vocab_path = "test_fixtures/roberta_model/vocabulary"
        self.vocab = Vocabulary.from_files(vocab_path)
        self.model_name = "roberta-base"
        self.token_indexer = PretrainedBertIndexer(
            pretrained_model=self.model_name,
            do_lowercase=False,
            max_pieces_per_token=5,
            special_tokens_fix=1,
        )

    def test_without_offsets(self):
        """Test input without offsets."""

        token_embedder = PretrainedBertEmbedder(
            pretrained_model=self.model_name,
            requires_grad=False,
            top_layer_only=True,
            special_tokens_fix=1,
        )
        input_ids = torch.LongTensor([[3, 5, 9, 1, 2], [1, 5, 0, 0, 0]])
        result = token_embedder(input_ids)

        assert list(result.shape) == [2, 5, 768]

    def test_without_offsets_special_tokens_fix_off(self):
        """Test input without offsets with special tokens fix off."""

        token_embedder = PretrainedBertEmbedder(
            pretrained_model=self.model_name,
            requires_grad=False,
            top_layer_only=True,
            special_tokens_fix=0,
        )
        input_ids = torch.LongTensor([[3, 5, 9, 1, 2], [1, 5, 0, 0, 0]])
        result = token_embedder(input_ids)

        assert list(result.shape) == [2, 5, 768]

    def test_with_offsets(self):
        """Test input with offsets."""

        token_embedder = PretrainedBertEmbedder(
            pretrained_model=self.model_name,
            requires_grad=False,
            top_layer_only=True,
            special_tokens_fix=1,
        )

        input_ids = torch.LongTensor([[3, 5, 9, 1, 2], [1, 5, 0, 0, 0]])
        offsets = torch.LongTensor([[0, 2, 4], [1, 0, 0]])

        result = token_embedder(input_ids, offsets=offsets)

        assert list(result.shape) == [2, 3, 768]

    def test_with_offsets_special_tokens_fix_off(self):
        """Test input with offsets with special tokens fix off."""

        token_embedder = PretrainedBertEmbedder(
            pretrained_model=self.model_name,
            requires_grad=False,
            top_layer_only=True,
            special_tokens_fix=0,
        )

        input_ids = torch.LongTensor([[3, 5, 9, 1, 2], [1, 5, 0, 0, 0]])
        offsets = torch.LongTensor([[0, 2, 4], [1, 0, 0]])

        result = token_embedder(input_ids, offsets=offsets)

        assert list(result.shape) == [2, 3, 768]

    def test_end_to_end(self):
        """Test token embedder end-to-end."""

        sentence1 = "the quickest quick brown fox jumped over the lazy dog"
        tokens1 = [Token(word) for word in sentence1.split()]

        sentence2 = "the quick brown fox jumped over the laziest lazy elmo"
        tokens2 = [Token(word) for word in sentence2.split()]

        instance1 = Instance(
            {"tokens": TextField(tokens1, {"bert": self.token_indexer})}
        )
        instance2 = Instance(
            {"tokens": TextField(tokens2, {"bert": self.token_indexer})}
        )

        batch = Batch([instance1, instance2])
        batch.index_instances(self.vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]["bert"]

        assert tokens["bert"].tolist() == [
            [627, 29155, 2119, 6219, 23602, 4262, 81, 5, 22414, 2335, 0, 0],
            [
                627,
                2119,
                6219,
                23602,
                4262,
                81,
                5,
                40154,
                7098,
                22414,
                1615,
                4992,
            ],
        ]

        assert tokens["bert-offsets"].tolist() == [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
        ]

        token_embedder = PretrainedBertEmbedder(
            self.model_name,
            requires_grad=False,
            top_layer_only=False,
            special_tokens_fix=1,
        )

        bert_vectors = token_embedder(tokens["bert"])
        assert list(bert_vectors.shape) == [2, 12, 768]

        # Offsets, should get 10 vectors back.
        bert_vectors = token_embedder(
            tokens["bert"], offsets=tokens["bert-offsets"]
        )
        assert list(bert_vectors.shape) == [2, 10, 768]

        # Now try top_layer_only = True
        tlo_embedder = PretrainedBertEmbedder(
            self.model_name, top_layer_only=True, special_tokens_fix=1
        )
        bert_vectors = tlo_embedder(tokens["bert"])
        assert list(bert_vectors.shape) == [2, 12, 768]

        bert_vectors = tlo_embedder(
            tokens["bert"], offsets=tokens["bert-offsets"]
        )
        assert list(bert_vectors.shape) == [2, 10, 768]

    def test_max_length(self):
        """Test that max input length works (default max len = 512)."""

        token_embedder = PretrainedBertEmbedder(
            self.model_name,
            requires_grad=False,
            top_layer_only=True,
            special_tokens_fix=1,
        )

        sentence = "the " * 512
        tokens = [Token(word) for word in sentence.split()]

        instance = Instance(
            {"tokens": TextField(tokens, {"bert": self.token_indexer})}
        )

        batch = Batch([instance])
        batch.index_instances(self.vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]["bert"]
        token_embedder(tokens["bert"], tokens["bert-offsets"])

    def test_max_length_raise_error(self):
        """Test that input greater than max length (default = 512) raises error."""

        token_embedder = PretrainedBertEmbedder(
            self.model_name,
            requires_grad=False,
            top_layer_only=True,
            special_tokens_fix=1,
        )

        sentence = "the " * 514
        tokens = [Token(word) for word in sentence.split()]

        instance = Instance(
            {"tokens": TextField(tokens, {"bert": self.token_indexer})}
        )

        batch = Batch([instance])
        batch.index_instances(self.vocab)

        padding_lengths = batch.get_padding_lengths()
        tensor_dict = batch.as_tensor_dict(padding_lengths)
        tokens = tensor_dict["tokens"]["bert"]
        with pytest.raises(IndexError):
            token_embedder(tokens["bert"], tokens["bert-offsets"])
