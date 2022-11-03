import torch
import numpy as np

from allennlp.common.testing import ModelTestCase
from allennlp.data import Batch
from allennlp.data.fields import TextField
from allennlp.data.fields import MetadataField
from allennlp.nn.util import move_to_device
from allennlp.data import Instance
from allennlp.data.fields import MetadataField
from allennlp.data import Token
from allennlp.data import Vocabulary
from allennlp.nn.util import move_to_device

from gector.basic_field_embedder import BasicTextFieldEmbedder
from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.tokenizer_indexer import PretrainedBertIndexer
from gector.seq2labels_model import Seq2Labels


class TestSeq2Labels(ModelTestCase):
    """Test class for Seq2Labels model."""

    def setup_method(self):
        """Set up indexers, embedders and dataset."""

        super(TestSeq2Labels, self).setup_method()

        vocab_path = "test_fixtures/roberta_model/vocabulary"
        self.vocab = Vocabulary.from_files(vocab_path)
        self.model_name = "roberta-base"
        token_indexer = PretrainedBertIndexer(
            pretrained_model=self.model_name,
            do_lowercase=False,
            max_pieces_per_token=5,
            special_tokens_fix=1,
        )
        token_embedder = PretrainedBertEmbedder(
            pretrained_model=self.model_name,
            requires_grad=False,
            top_layer_only=True,
            special_tokens_fix=1,
        )

        embedders = {"bert": token_embedder}

        self.text_field_embedder = BasicTextFieldEmbedder(
            token_embedders=embedders,
            embedder_to_indexer_map={"bert": ["bert", "bert-offsets"]},
            allow_unmatched_keys=True,
        )

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.device_index = 0 if torch.cuda.is_available() else -1

        sentence1 = "the quickest quick brown fox jumped over the lazy dog"
        sentence1_words = [word for word in sentence1.split()]
        tokens1 = [Token(word) for word in ["$START"] + sentence1_words]

        sentence2 = "the quick brown fox jumped over the laziest lazy elmo"
        sentence2_words = [word for word in sentence2.split()]
        tokens2 = [Token(word) for word in ["$START"] + sentence2_words]

        instance1 = Instance(
            {
                "tokens": TextField(tokens1, {"bert": token_indexer}),
                "metadata": MetadataField({"words": sentence1_words}),
            }
        )
        instance2 = Instance(
            {
                "tokens": TextField(tokens2, {"bert": token_indexer}),
                "metadata": MetadataField({"words": sentence2_words}),
            }
        )

        self.batch = Batch([instance1, instance2])

    def test_forward_pass_runs_correctly(self):
        """Test if forward pass returns correct output."""

        self.batch.index_instances(self.vocab)
        padding_lengths = self.batch.get_padding_lengths()
        training_tensors = self.batch.as_tensor_dict(padding_lengths)
        training_tensors = move_to_device(training_tensors, self.device_index)
        model = Seq2Labels(
            vocab=self.vocab,
            text_field_embedder=self.text_field_embedder,
            confidence=0,
            del_confidence=0,
        ).to(self.device)

        output_dict = model(**training_tensors)
        output_dict = model.make_output_human_readable(output_dict)

        assert set(output_dict.keys()) == set(
            [
                "logits_labels",
                "logits_d_tags",
                "class_probabilities_labels",
                "class_probabilities_d_tags",
                "max_error_probability",
                "labels",
                "d_tags",
                "words",
                "corrected_words",
            ]
        )
        probs = output_dict["class_probabilities_labels"][0].data.cpu().numpy()

        np.testing.assert_almost_equal(np.sum(probs, -1), np.full((11), 1), 5)
