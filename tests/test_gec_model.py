from pathlib import Path
import torch
import requests
from tqdm import tqdm

from allennlp.common.testing import ModelTestCase
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import Token
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField
from allennlp.data import Batch

from gector.gec_model import GecBERTModel
from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.tokenizer_indexer import PretrainedBertIndexer


class TestGecModel(ModelTestCase):
    """Test class for GecModel"""

    def setup_method(self):
        super().setup_method()
        self.vocab_path = "test_fixtures/roberta_model/vocabulary"
        self.vocab = Vocabulary.from_files(self.vocab_path)
        self.model_name = "roberta-base"
        model_url = "https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th"
        test_fixtures_dir_path = Path(__file__).parent.parent / "test_fixtures"
        model_path = test_fixtures_dir_path / "roberta_model" / "weights.th"
        if not model_path.exists():
            response = requests.get(model_url)
            with model_path.open("wb") as out_fp:
                # Write out data with progress bar
                for data in tqdm(response.iter_content()):
                    out_fp.write(data)
        assert model_path.exists()
        self.model_path = model_path

    def test_gec_model_prediction(self):
        """Test simple prediction with GecBERTModel"""
        gec_model = GecBERTModel(
            vocab_path=self.vocab_path,
            model_paths=[self.model_path],
            max_len=50,
            iterations=5,
            min_error_probability=0.0,
            lowercase_tokens=0,
            model_name="roberta",
            special_tokens_fix=1,
            log=False,
            confidence=0,
            del_confidence=0,
            is_ensemble=0,
            weights=None,
        )

        sentence1 = "I run to a stores every day."
        sentence2 = "the quick brown foxes jumps over a elmo's laziest friend"
        # This mimics how batches of requests are constructed in predict.py's predict_for_file function
        input_data = [sentence1, sentence2]
        input_data = [sentence.split() for sentence in input_data]
        final_batch, total_updates = gec_model.handle_batch(input_data)
        # subject verb agreement is not fixed in the second sentence when predicting using GecModel
        # (i.e.) brown foxes jump
        assert final_batch == [
            ["I", "run", "to", "the", "stores", "every", "day."],
            [
                "The",
                "quick",
                "brown",
                "foxes",
                "jumps",
                "over",
                "Elmo's",
                "laziest",
                "friend",
            ],
        ]
        assert total_updates == 2

    def test_gec_model_problem_prediction(self):
        """Test problem prediction with GecBERTModel"""
        gec_model = GecBERTModel(
            vocab_path=self.vocab_path,
            model_paths=[self.model_path],
            max_len=50,
            min_len=3,
            iterations=5,
            min_error_probability=0.0,
            lowercase_tokens=0,
            model_name="roberta",
            special_tokens_fix=1,
            log=False,
            confidence=0,
            del_confidence=0,
            is_ensemble=0,
            weights=None,
        )

        sentence1 = "everyday on tv , there are always the entertainers and athletes dedicating their effort and abilities to entertain audiences ."
        sentence2 = "therefore , in return , these people always hope to earn milionsof dollars every year ."
        sentence3 = "they all have worked hard to present people their best of the best ."
        sentence4 = "therefore , i think some famous althletes and entertainers earn millions of dollars every year is fair , they deserve it ."
        sentence5 = "If a gene runs in the family , one of the family member test positive , whom does he need to tell ."
        sentence6 = "And if you still decide to have the baby , since the technology has been developed so advanced , it might be possible in the future that the application of altering gene is perfected and widely used , you can then choose to give birth to babies by giving them a brighter future ."

        input_data = [
            sentence1,
            sentence2,
            sentence3,
            sentence4,
            sentence5,
            sentence6,
        ]
        # This mimics how batches of requests are constructed in predict.py's predict_for_file function
        input_data = [sentence.split() for sentence in input_data]
        final_batch, total_updates = gec_model.handle_batch(input_data)

        assert final_batch == [
            [
                "On",
                "TV",
                ",",
                "there",
                "are",
                "always",
                "entertainers",
                "and",
                "athletes",
                "dedicating",
                "their",
                "effort",
                "and",
                "abilities",
                "to",
                "entertaining",
                "audiences",
                ".",
            ],
            [
                "Therefore",
                ",",
                "in",
                "return",
                ",",
                "these",
                "people",
                "always",
                "hope",
                "to",
                "earn",
                "milionsof",
                "dollars",
                "every",
                "year",
                ".",
            ],
            [
                "They",
                "all",
                "worked",
                "hard",
                "to",
                "give",
                "people",
                "the",
                "best",
                "of",
                "the",
                "best",
                ".",
            ],
            [
                "Therefore",
                ",",
                "I",
                "think",
                "some",
                "famous",
                "althletes",
                "and",
                "entertainers",
                "earning",
                "millions",
                "of",
                "dollars",
                "every",
                "year",
                "is",
                "fair",
                ".",
                "They",
                "deserve",
                "it",
                ".",
            ],
            [
                "If",
                "a",
                "gene",
                "runs",
                "in",
                "the",
                "family",
                ",",
                "one",
                "of",
                "the",
                "family",
                "members",
                "tests",
                "positive",
                ",",
                "does",
                "he",
                "need",
                "to",
                "tell",
                "?",
            ],
            [
                "And",
                "if",
                "you",
                "still",
                "decide",
                "to",
                "have",
                "a",
                "baby",
                ",",
                "since",
                "the",
                "technology",
                "has",
                "been",
                "developed",
                ",",
                "it",
                "might",
                "be",
                "possible",
                "in",
                "the",
                "future",
                "that",
                "the",
                "application",
                "of",
                "altering",
                "genes",
                "be",
                "perfected",
                "and",
                "widely",
                "used",
                ".",
                "You",
                "can",
                "then",
                "choose",
                "to",
                "give",
                "birth",
                "to",
                "babies",
                "by",
                "giving",
                "them",
                "a",
                "brighter",
                "future",
                ".",
            ],
        ]
        assert total_updates == 17
