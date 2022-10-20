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
import torch
from allennlp.nn import util
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token


@Predictor.register("gec-predictor")
class GecPredictor(Predictor):
    """
    A Predictor for generating predictions from GECToR.

    Note that currently, this is unable to handle ensemble predictions.
    """

    def __init__(
        self, model: Model, dataset_reader: DatasetReader, iterations: int = 5
    ) -> None:
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
        return self.predict_batch_json(
            [{"sentence": sentence} for sentence in sentences]
        )

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
    def predict_batch_instance(
        self, instances: List[Instance]
    ) -> List[JsonDict]:
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

        # Make deep copy of batch
        final_batch = instances[:]

        final_outputs = [None] * len(instances)

        prev_preds_dict = {
            id: [final_batch[id].fields["tokens"].tokens]
            for id in range(len(final_batch))
        }

        pred_ids = [id for id in range(len(instances))]

        for iter in range(self._iterations):

            if len(pred_ids) == 0:
                break

            orig_batch = [final_batch[pred_id] for pred_id in pred_ids]
            outputs = self._model.forward_on_instances(orig_batch)
            new_pred_ids = []
            for op_ind, pred_id in enumerate(pred_ids):
                final_outputs[pred_id] = outputs[op_ind]
                orig = final_batch[pred_id]
                tokens = [
                    Token(word)
                    for word in ["$START"] + outputs[op_ind]["corrected_words"]
                ]
                pred = self._dataset_reader.text_to_instance(tokens)
                prev_preds = prev_preds_dict[pred_id]
                if (
                    orig.fields["tokens"].tokens != pred.fields["tokens"].tokens
                    and pred.fields["tokens"].tokens not in prev_preds
                ):
                    # Set sentence instance to predicted sent in batch
                    final_batch[pred_id] = pred
                    # Append orig_id in new_pred_ids
                    new_pred_ids.append(pred_id)
                    # Append new prediction to prev_preds_dict
                    prev_preds_dict[pred_id].append(
                        pred.fields["tokens"].tokens
                    )
                elif (
                    orig.fields["tokens"].tokens != pred.fields["tokens"].tokens
                    and pred.fields["tokens"].tokens in prev_preds
                ):
                    # update final batch, but stop iterations
                    final_batch[pred_id] = pred
                else:
                    continue
            pred_ids = new_pred_ids
        return sanitize(final_outputs)

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
