"""Tweaked version of corresponding AllenNLP file"""
import logging
from collections import defaultdict
from typing import Dict, List, Callable

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides
from transformers import AutoTokenizer

from utils.helpers import START_TOKEN

from gector.tokenization import tokenize_batch
import copy

logger = logging.getLogger(__name__)


# TODO(joelgrus): Figure out how to generate token_type_ids out of this token indexer.


class TokenizerIndexer(TokenIndexer[int]):
    """
    A token indexer that does the wordpiece-tokenization (e.g. for BERT embeddings).
    If you are using one of the pretrained BERT models, you'll want to use the ``PretrainedBertIndexer``
    subclass rather than this base class.

    Parameters
    ----------
    tokenizer : ``Callable[[str], List[str]]``
        A function that does the actual tokenization.
    max_pieces : int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Any inputs longer than this will
        either be truncated (default), or be split apart and batched using a
        sliding window.
    token_min_padding_length : ``int``, optional (default=``0``)
        See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Callable[[str], List[str]],
                 max_pieces: int = 512,
                 max_pieces_per_token: int = 3,
                 token_min_padding_length: int = 0) -> None:
        super().__init__(token_min_padding_length)

        # The BERT code itself does a two-step tokenization:
        #    sentence -> [words], and then word -> [wordpieces]
        # In AllenNLP, the first step is implemented as the ``BertBasicWordSplitter``,
        # and this token indexer handles the second.

        self.tokenizer = tokenizer
        self.max_pieces_per_token = max_pieces_per_token
        self.max_pieces = max_pieces
        self.max_pieces_per_sentence = 80

    @overrides
    def tokens_to_indices(self, tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        text = [token.text for token in tokens]
        batch_tokens = [text]

        output_fast = tokenize_batch(self.tokenizer,
                                     batch_tokens,
                                     max_bpe_length=self.max_pieces,
                                     max_bpe_pieces=self.max_pieces_per_token)
        output_fast = {k: v[0] for k, v in output_fast.items()}
        return output_fast

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    @overrides
    def get_padding_token(self) -> int:
        return 0

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[int]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:  # pylint: disable=unused-argument
        return {key: pad_sequence_to_length(val, desired_num_tokens[key])
                for key, val in tokens.items()}

    @overrides
    def get_keys(self, index_name: str) -> List[str]:
        """
        We need to override this because the indexer generates multiple keys.
        """
        # pylint: disable=no-self-use
        return [index_name, f"{index_name}-offsets", f"{index_name}-type-ids", "mask"]


class PretrainedBertIndexer(TokenizerIndexer):
    # pylint: disable=line-too-long
    """
    A ``TokenIndexer`` corresponding to a pretrained BERT model.

    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .txt file with its vocabulary.
        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py#L33
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    do_lowercase: ``bool``, optional (default = True)
        Whether to lowercase the tokens before converting to wordpiece ids.
    max_pieces: int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Any inputs longer than this will
        either be truncated (default), or be split apart and batched using a
        sliding window.
    """

    def __init__(self,
                 pretrained_model: str,
                 do_lowercase: bool = True,
                 max_pieces: int = 512,
                 max_pieces_per_token: int = 5,
                 special_tokens_fix: int = 0) -> None:

        if pretrained_model.endswith("-cased") and do_lowercase:
            logger.warning("Your BERT model appears to be cased, "
                           "but your indexer is lowercasing tokens.")
        elif pretrained_model.endswith("-uncased") and not do_lowercase:
            logger.warning("Your BERT model appears to be uncased, "
                           "but your indexer is not lowercasing tokens.")

        model_name = copy.deepcopy(pretrained_model)

        model_tokenizer = AutoTokenizer.from_pretrained(
            model_name, do_lower_case=do_lowercase, do_basic_tokenize=False, use_fast=True)

        # to adjust all tokenizers
        if hasattr(model_tokenizer, 'encoder'):
            model_tokenizer.vocab = model_tokenizer.encoder
        if hasattr(model_tokenizer, 'sp_model'):
            model_tokenizer.vocab = defaultdict(lambda: 1)
            for i in range(model_tokenizer.sp_model.get_piece_size()):
                model_tokenizer.vocab[model_tokenizer.sp_model.id_to_piece(i)] = i

        if special_tokens_fix:
            model_tokenizer.add_tokens([START_TOKEN])
            model_tokenizer.vocab[START_TOKEN] = len(model_tokenizer) - 1

        super().__init__(tokenizer=model_tokenizer,
                         max_pieces=max_pieces,
                         max_pieces_per_token=max_pieces_per_token
                        )

