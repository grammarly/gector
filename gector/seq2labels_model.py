"""Basic model. Predicts tags for every token"""
from typing import Dict, Optional, List, Any

import numpy
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
from torch.nn.modules.linear import Linear


@Model.register("seq2labels")
class Seq2Labels(Model):
    """
    This ``Seq2Labels`` simply encodes a sequence of text with a stacked ``Seq2SeqEncoder``, then
    predicts a tag (or couple tags) for each token in the sequence.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    calculate_span_f1 : ``bool``, optional (default=``None``)
        Calculate span-level F1 metrics during training. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    label_encoding : ``str``, optional (default=``None``)
        Label encoding to use when calculating span f1.
        Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if ``calculate_span_f1`` is true.
    label_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric, if desired.
        Unless you did something unusual, the default value should be what you want.
    verbose_metrics : ``bool``, optional (default = False)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 predictor_dropout=0.0,
                 labels_namespace: str = "labels",
                 detect_namespace: str = "d_tags",
                 verbose_metrics: bool = False,
                 label_smoothing: float = 0.0,
                 confidence: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Seq2Labels, self).__init__(vocab, regularizer)

        self.label_namespaces = [labels_namespace,
                                 detect_namespace]
        self.text_field_embedder = text_field_embedder
        self.num_labels_classes = self.vocab.get_vocab_size(labels_namespace)
        self.num_detect_classes = self.vocab.get_vocab_size(detect_namespace)
        self.label_smoothing = label_smoothing
        self.confidence = confidence
        self.incorr_index = self.vocab.get_token_index("INCORRECT",
                                                       namespace=detect_namespace)

        self._verbose_metrics = verbose_metrics
        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(predictor_dropout))

        self.tag_labels_projection_layer = TimeDistributed(
            Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_labels_classes))

        self.tag_detect_projection_layer = TimeDistributed(
            Linear(text_field_embedder._token_embedders['bert'].get_output_dim(), self.num_detect_classes))

        self.metrics = {"accuracy": CategoricalAccuracy()}

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                d_tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        lables : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        d_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containing the original words in the sentence to be tagged under a 'words' key.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        encoded_text = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = encoded_text.size()
        mask = get_text_field_mask(tokens)
        logits_labels = self.tag_labels_projection_layer(self.predictor_dropout(encoded_text))
        logits_d = self.tag_detect_projection_layer(encoded_text)

        class_probabilities_labels = F.softmax(logits_labels, dim=-1).view(
            [batch_size, sequence_length, self.num_labels_classes])
        class_probabilities_d = F.softmax(logits_d, dim=-1).view(
            [batch_size, sequence_length, self.num_detect_classes])
        error_probs = class_probabilities_d[:, :, self.incorr_index] * mask
        incorr_prob = torch.max(error_probs, dim=-1)[0]

        if self.confidence > 0:
            probability_change = [self.confidence] + [0] * (self.num_labels_classes - 1)
            class_probabilities_labels += torch.cuda.FloatTensor(probability_change).repeat(
                (batch_size, sequence_length, 1))

        output_dict = {"logits_labels": logits_labels,
                       "logits_d_tags": logits_d,
                       "class_probabilities_labels": class_probabilities_labels,
                       "class_probabilities_d_tags": class_probabilities_d,
                       "max_error_probability": incorr_prob}
        if labels is not None and d_tags is not None:
            loss_labels = sequence_cross_entropy_with_logits(logits_labels, labels, mask,
                                                             label_smoothing=self.label_smoothing)
            loss_d = sequence_cross_entropy_with_logits(logits_d, d_tags, mask)
            for metric in self.metrics.values():
                metric(logits_labels, labels, mask.float())
                metric(logits_d, d_tags, mask.float())
            output_dict["loss"] = loss_labels + loss_d

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        for label_namespace in self.label_namespaces:
            all_predictions = output_dict[f'class_probabilities_{label_namespace}']
            all_predictions = all_predictions.cpu().data.numpy()
            if all_predictions.ndim == 3:
                predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
            else:
                predictions_list = [all_predictions]
            all_tags = []

            for predictions in predictions_list:
                argmax_indices = numpy.argmax(predictions, axis=-1)
                tags = [self.vocab.get_token_from_index(x, namespace=label_namespace)
                        for x in argmax_indices]
                all_tags.append(tags)
            output_dict[f'{label_namespace}'] = all_tags
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}
        return metrics_to_return
