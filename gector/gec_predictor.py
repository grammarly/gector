from typing import Dict, List

import numpy
from allennlp.predictors import Predictor
from allennlp.models import Model
from allennlp.common.util import sanitize
from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.models import Model
from utils.helpers import START_TOKEN


@Predictor.register("gec-predictor")
class GecPredictor(Predictor):
    """
    A Predictor for generating predictions from GECToR.

    Note that currently, this is unable to handle ensemble predictions.
    """

    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 iterations: int = 3) -> None:
        """
        Parameters
        ---------
        model: Model
            An instantiated `Seq2Labels` model for performing grammatical error correction.
        dataset_reader: DatasetReader
            An instantiated dataset reader, typically `Seq2LabelsDatasetReader`.
        iterations: int
            This represents the number of times grammatical error correction is applied to the input.
        """
        super().__init__(model, dataset_reader)
        self._tokenizer = JustSpacesWordSplitter()
        self._iterations = iterations

    def predict(self, sentence: str) -> JsonDict:
        """
        Generate error correction predictions for a single input string.

        Parameters
        ----------
        sentence: str
            The input text to perform error correction on.

        Returns
        -------
        JsonDict
            A dictionary containing the following keys:
                - logits_labels
                - logits_d_tags
                - class_probabilities_labels
                - class_probabilities_d_tags
                - max_error_probability
                - words
                - labels
                - d_tags
                - corrected_words
            For an explanation of each of these see `Seq2Labels.decode()`.
        """
        return self.predict_json({"sentence": sentence})

    def predict_batch(self, sentences: List[str]) -> List[JsonDict]:
        """
        Generate predictions for a sequence of input strings.

        Parameters
        ----------
        sentences: List[str]
            A list of strings to correct.

        Returns
        -------
        List[JsonDict]
            A list of dictionaries, each containing the following keys:
                - logits_labels
                - logits_d_tags
                - class_probabilities_labels
                - class_probabilities_d_tags
                - max_error_probability
                - words
                - labels
                - d_tags
                - corrected_words
            For an explanation of each of these see `Seq2Labels.decode()`.
        """
        return self.predict_batch_json([{"sentence": sentence} for sentence in sentences])

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        """
        This special predict_instance method allows for applying the correction model multiple times.

        Parameters
        ---------

        Returns
        -------
        JsonDict
            A dictionary containing the following keys:
                - logits_labels
                - logits_d_tags
                - class_probabilities_labels
                - class_probabilities_d_tags
                - max_error_probability
                - words
                - labels
                - d_tags
                - corrected_words
            For an explanation of each of these see `Seq2Labels.decode()`.
        """
        for i in range(self._iterations):
            output = self._model.forward_on_instance(instance)
            # integrate predictions back into instance for next iteration
            tokens = [Token(word) for word in output["corrected_words"]]
            instance = self._dataset_reader.text_to_instance(tokens)
        return sanitize(output)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        """
        This special predict_batch_instance method allows for applying the correction model multiple times.

        Parameters
        ----------

        Returns
        -------
        List[JsonDict]
            A list of dictionaries, each containing the following keys:
                - logits_labels
                - logits_d_tags
                - class_probabilities_labels
                - class_probabilities_d_tags
                - max_error_probability
                - words
                - labels
                - d_tags
                - corrected_words
            For an explanation of each of these see `Seq2Labels.decode()`.
        """
        for i in range(self._iterations):
            outputs = self._model.forward_on_instances(instances)
            corrected_instances = []
            for output in outputs:
                tokens = [Token(word) for word in output["corrected_words"]]
                instance = self._dataset_reader.text_to_instance(tokens)
                corrected_instances.append(instance)
            instances = corrected_instances
        return sanitize(outputs)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Convert a JsonDict into an Instance.

        This is used internally by `self.predict_json()`.

        Parameters
        ----------
        json_dict: JsonDict
            Expects a dict with a single key "sentence" with a value representing the string to correct.
            i.e. ``{"sentence": "..."}``.

        Returns
        ------
        Instance
            An instance with the following fields:
                - tokens
                - metadata
                - labels
                - d_tags
        """
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.split_words(sentence)
        # Add start token to front
        tokens = [Token(START_TOKEN)] + tokens
        return self._dataset_reader.text_to_instance(tokens)
