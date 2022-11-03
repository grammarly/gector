from typing import Dict, List

import numpy
from allennlp.predictors import Predictor
from allennlp.models import Model
from allennlp.common.util import sanitize
from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers import WhitespaceTokenizer 
from allennlp.models import Model

from gector.utils.helpers import START_TOKEN


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
        #self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)
        self._tokenizer = WhitespaceTokenizer()
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
        instance: Instance
            Instance to be predicted

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
        return self.predict_batch_instance([instance])[0]

    @overrides
    def predict_batch_instance(
        self, instances: List[Instance]
    ) -> List[JsonDict]:
        """
        This special predict_batch_instance method allows for applying the correction model multiple times.

        Parameters
        ----------
        instances: List[Instance]
            Instances to be predicted

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

        # Create list to store final predictions
        final_outputs = [None] * len(instances)

        # This dictionary keeps track of predictions made in every iteration
        prev_preds_dict = {}

        # This list contains IDs of sentences to be passed into model
        pred_ids = []

        # Populating `prev_preds_dict` and `pred_ids`
        for id, instance in enumerate(final_batch):
            prev_preds_dict[id] = [instance.fields["tokens"].tokens]
            # If len(tokens) is less than 4 ($START + 3 tokens)
            # we will not correct it.
            # It is directly written to output.
            if len(instance.fields["tokens"].tokens) < 4:
                final_outputs[id] = {
                    "logits_labels": None,
                    "logits_d_tags": None,
                    "class_probabilities_labels": None,
                    "class_probabilities_d_tags": None,
                    "max_error_probability": None,
                    "words": instance.fields["tokens"].tokens[1:],
                    "labels": None,
                    "d_tags": None,
                    "corrected_words": instance.fields["tokens"].tokens[1:],
                }
            else:
                pred_ids.append(id)

        # Applying correction model multiple times
        for _ in range(self._iterations):

            # If no sentences need to be passed into model
            if len(pred_ids) == 0:
                break

            # Create batch of instances to be passed into model
            orig_batch = [final_batch[pred_id] for pred_id in pred_ids]

            # Pass into model
            outputs = self._model.forward_on_instances(orig_batch)

            new_pred_ids = []

            # Output_ID and Pred_ID in pred_ids
            for op_ind, pred_id in enumerate(pred_ids):

                # Update final outputs
                final_outputs[pred_id] = outputs[op_ind]
                orig = final_batch[pred_id]

                # Create tokens from corrected words for next iter
                tokens = [
                    Token(word)
                    for word in ["$START"] + outputs[op_ind]["corrected_words"]
                ]

                # Tokens to instance
                pred = self._dataset_reader.text_to_instance(tokens)
                prev_preds = prev_preds_dict[pred_id]

                # If model output is different from previous iter outputs
                # Update input batch, append to dict and add to `pred_ids`
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
                # If model output is same as that in prev iter, update final batch
                # but stop passing it into the model for future iters
                # This means that no corrections have been made in this iteration
                elif (
                    orig.fields["tokens"].tokens != pred.fields["tokens"].tokens
                    and pred.fields["tokens"].tokens in prev_preds
                ):
                    # update final batch, but stop iterations
                    final_batch[pred_id] = pred
                else:
                    continue

            # Update `pred_ids` with new indices to be predicted
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
        tokens = self._tokenizer.tokenize(sentence)
        # Add start token to front
        tokens = [Token(START_TOKEN)] + tokens
        return self._dataset_reader.text_to_instance(tokens)

    @overrides
    def predictions_to_labeled_instances(self,
                                         instance: Instance,
                                         outputs: Dict[str, numpy.ndarray]) -> List[Instance]:
        """
        This method creates an instance out of the predictions generated by the model.
        """
        NotImplemented
