from pathlib import Path
import torch
import requests
from tqdm import tqdm

from allennlp.common.testing import ModelTestCase
from allennlp.predictors import Predictor
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import Token
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField
from allennlp.data.dataset import Batch

from gector.gec_predictor import GecPredictor
from gector.gec_model import GecBERTModel
from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.tokenizer_indexer import PretrainedBertIndexer

# These imports are required so that instantiating the predictor can be done automatically
from gector.datareader import Seq2LabelsDatasetReader
from gector.seq2labels_model import Seq2Labels
from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.tokenizer_indexer import PretrainedBertIndexer




class TestGecPredictor(ModelTestCase):
    """Test class for GecModel"""

    def setUp(self):
        super().setUp()
        test_fixtures_dir_path = Path(__file__).parent.parent / "test_fixtures"
        # Download weights for model archive
        weights_url = "https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th"
        test_fixtures_dir_path = Path(__file__).parent.parent / "test_fixtures"
        model_path = test_fixtures_dir_path / "roberta_model" / "weights.th"
        if not model_path.exists():
            response = requests.get(weights_url)
            with model_path.open("wb") as out_fp:
                # Write out data with progress bar
                for data in tqdm(response.iter_content()):
                    out_fp.write(data)
        model_path = test_fixtures_dir_path / "roberta_model"
        self.model_path = model_path
        sentence1 = "I run to a stores every day."
        sentence2 = "the quick brown foxes jumps over a elmo's laziest friend"
        # This micmics how batches of requests are constructed in predict.py's predict_for_file function
        self.input_data = [sentence1, sentence2]

    def test_gec_predictor_prediction(self):
        """Test simple prediction integration test with GecPredictor"""
        gec_model = Predictor.from_path(self.model_path, predictor_name="gec-predictor")
        prediction = gec_model.predict_batch(self.input_data)
        # subject verb agreement is not fixed in the second sentence when predicting using GecModel
        # (i.e.) brown foxes jump
        assert set(prediction[0].keys()) == {'logits_labels', 'logits_d_tags', 'class_probabilities_labels',
                                             'class_probabilities_d_tags', 'max_error_probability', 'words', 'labels',
                                             'd_tags', 'corrected_words'}
        assert prediction[0]["corrected_words"] == ['I', 'run', 'to', 'the', 'stores', 'every', 'day.']
        assert prediction[1]["corrected_words"] == ['The', 'quick', 'brown', 'foxes', 'jumps', 'over', "Elmo's",
                                                    'laziest', 'friend']
