import argparse
import os
from random import seed

import torch
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.datareader import Seq2LabelsDatasetReader
from gector.seq2labels_model import Seq2Labels
from gector.trainer import Trainer
from gector.wordpiece_indexer import PretrainedBertIndexer
from utils.helpers import get_weights_name


def fix_seed():
    torch.manual_seed(1)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed(43)


def get_token_indexers(model_name, max_pieces_per_token=5, lowercase_tokens=True, special_tokens_fix=0, is_test=False):
    bert_token_indexer = PretrainedBertIndexer(
        pretrained_model=model_name,
        max_pieces_per_token=max_pieces_per_token,
        do_lowercase=lowercase_tokens,
        use_starting_offsets=True,
        special_tokens_fix=special_tokens_fix,
        is_test=is_test
    )
    return {'bert': bert_token_indexer}


def get_token_embedders(model_name, tune_bert=False, special_tokens_fix=0):
    take_grads = True if tune_bert > 0 else False
    bert_token_emb = PretrainedBertEmbedder(
        pretrained_model=model_name,
        top_layer_only=True, requires_grad=take_grads,
        special_tokens_fix=special_tokens_fix)

    token_embedders = {'bert': bert_token_emb}
    embedder_to_indexer_map = {"bert": ["bert", "bert-offsets"]}

    text_filed_emd = BasicTextFieldEmbedder(token_embedders=token_embedders,
                                            embedder_to_indexer_map=embedder_to_indexer_map,
                                            allow_unmatched_keys=True)
    return text_filed_emd


def get_data_reader(model_name, max_len, skip_correct=False, skip_complex=0,
                    test_mode=False, tag_strategy="keep_one",
                    broken_dot_strategy="keep", lowercase_tokens=True,
                    max_pieces_per_token=3, tn_prob=0, tp_prob=1, special_tokens_fix=0,):
    token_indexers = get_token_indexers(model_name,
                                        max_pieces_per_token=max_pieces_per_token,
                                        lowercase_tokens=lowercase_tokens,
                                        special_tokens_fix=special_tokens_fix,
                                        is_test=test_mode)
    reader = Seq2LabelsDatasetReader(token_indexers=token_indexers,
                                     max_len=max_len,
                                     skip_correct=skip_correct,
                                     skip_complex=skip_complex,
                                     test_mode=test_mode,
                                     tag_strategy=tag_strategy,
                                     broken_dot_strategy=broken_dot_strategy,
                                     lazy=True,
                                     tn_prob=tn_prob,
                                     tp_prob=tp_prob)
    return reader


def get_model(model_name, vocab, tune_bert=False,
              predictor_dropout=0,
              label_smoothing=0.0,
              confidence=0,
              special_tokens_fix=0):
    token_embs = get_token_embedders(model_name, tune_bert=tune_bert, special_tokens_fix=special_tokens_fix)
    model = Seq2Labels(vocab=vocab,
                       text_field_embedder=token_embs,
                       predictor_dropout=predictor_dropout,
                       label_smoothing=label_smoothing,
                       confidence=confidence)
    return model


def main(args):
    fix_seed()
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    weights_name = get_weights_name(args.transformer_model, args.lowercase_tokens)
    # read datasets
    reader = get_data_reader(weights_name, args.max_len, skip_correct=bool(args.skip_correct),
                             skip_complex=args.skip_complex,
                             test_mode=False,
                             tag_strategy=args.tag_strategy,
                             lowercase_tokens=args.lowercase_tokens,
                             max_pieces_per_token=args.pieces_per_token,
                             tn_prob=args.tn_prob,
                             tp_prob=args.tp_prob,
                             special_tokens_fix=args.special_tokens_fix)
    train_data = reader.read(args.train_set)
    dev_data = reader.read(args.dev_set)

    default_tokens = [DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN]
    namespaces = ['labels', 'd_tags']
    tokens_to_add = {x: default_tokens for x in namespaces}
    # build vocab
    if args.vocab_path:
        vocab = Vocabulary.from_files(args.vocab_path)
    else:
        vocab = Vocabulary.from_instances(train_data,
                                          max_vocab_size={'tokens': 30000,
                                                          'labels': args.target_vocab_size,
                                                          'd_tags': 2},
                                          tokens_to_add=tokens_to_add)
    vocab.save_to_files(os.path.join(args.model_dir, 'vocabulary'))

    print("Data is loaded")
    model = get_model(weights_name, vocab,
                      tune_bert=args.tune_bert,
                      predictor_dropout=args.predictor_dropout,
                      label_smoothing=args.label_smoothing,
                      special_tokens_fix=args.special_tokens_fix)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            cuda_device = list(range(torch.cuda.device_count()))
        else:
            cuda_device = 0
    else:
        cuda_device = -1

    if args.pretrain:
        model.load_state_dict(torch.load(os.path.join(args.pretrain_folder, args.pretrain + '.th')))

    model = model.to(device)

    print("Model is set")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10)
    instances_per_epoch = None if not args.updates_per_epoch else \
        int(args.updates_per_epoch * args.batch_size * args.accumulation_size)
    iterator = BucketIterator(batch_size=args.batch_size,
                              sorting_keys=[("tokens", "num_tokens")],
                              biggest_batch_first=True,
                              max_instances_in_memory=args.batch_size * 20000,
                              instances_per_epoch=instances_per_epoch,
                              )
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      iterator=iterator,
                      train_dataset=train_data,
                      validation_dataset=dev_data,
                      serialization_dir=args.model_dir,
                      patience=args.patience,
                      num_epochs=args.n_epoch,
                      cuda_device=cuda_device,
                      shuffle=False,
                      accumulated_batch_count=args.accumulation_size,
                      cold_step_count=args.cold_steps_count,
                      cold_lr=args.cold_lr,
                      cuda_verbose_step=int(args.cuda_verbose_steps)
                      if args.cuda_verbose_steps else None
                      )
    print("Start training")
    trainer.train()

    # Here's how to save the model.
    out_model = os.path.join(args.model_dir, 'model.th')
    with open(out_model, 'wb') as f:
        torch.save(model.state_dict(), f)
    print("Model is dumped")


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set',
                        help='Path to the train data', required=True)
    parser.add_argument('--dev_set',
                        help='Path to the dev data', required=True)
    parser.add_argument('--model_dir',
                        help='Path to the model dir', required=True)
    parser.add_argument('--vocab_path',
                        help='Path to the model vocabulary directory.'
                             'If not set then build vocab from data',
                        default='')
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of the batch.',
                        default=32)
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--target_vocab_size',
                        type=int,
                        help='The size of target vocabularies.',
                        default=1000)
    parser.add_argument('--n_epoch',
                        type=int,
                        help='The number of epoch for training model.',
                        default=20)
    parser.add_argument('--patience',
                        type=int,
                        help='The number of epoch with any improvements'
                             ' on validation set.',
                        default=3)
    parser.add_argument('--skip_correct',
                        type=int,
                        help='If set than correct sentences will be skipped '
                             'by data reader.',
                        default=1)
    parser.add_argument('--skip_complex',
                        type=int,
                        help='If set than complex corrections will be skipped '
                             'by data reader.',
                        choices=[0, 1, 2, 3, 4, 5],
                        default=0)
    parser.add_argument('--tune_bert',
                        type=int,
                        help='If more then 0 then fine tune bert.',
                        default=1)
    parser.add_argument('--tag_strategy',
                        choices=['keep_one', 'merge_all'],
                        help='The type of the data reader behaviour.',
                        default='keep_one')
    parser.add_argument('--accumulation_size',
                        type=int,
                        help='How many batches do you want accumulate.',
                        default=4)
    parser.add_argument('--lr',
                        type=float,
                        help='Set initial learning rate.',
                        default=1e-5)
    parser.add_argument('--cold_steps_count',
                        type=int,
                        help='Whether to train only classifier layers first.',
                        default=4)
    parser.add_argument('--cold_lr',
                        type=float,
                        help='Learning rate during cold_steps.',
                        default=1e-3)
    parser.add_argument('--predictor_dropout',
                        type=float,
                        help='The value of dropout for predictor.',
                        default=0.0)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--pieces_per_token',
                        type=int,
                        help='The max number for pieces per token.',
                        default=5)
    parser.add_argument('--cuda_verbose_steps',
                        help='Number of steps after which CUDA memory information is printed. '
                             'Makes sense for local testing. Usually about 1000.',
                        default=None)
    parser.add_argument('--label_smoothing',
                        type=float,
                        help='The value of parameter alpha for label smoothing.',
                        default=0.0)
    parser.add_argument('--tn_prob',
                        type=float,
                        help='The probability to take TN from data.',
                        default=0)
    parser.add_argument('--tp_prob',
                        type=float,
                        help='The probability to take TP from data.',
                        default=1)
    parser.add_argument('--updates_per_epoch',
                        type=int,
                        help='If set then each epoch will contain the exact amount of updates.',
                        default=0)
    parser.add_argument('--pretrain_folder',
                        help='The name of the pretrain folder.')
    parser.add_argument('--pretrain',
                        help='The name of the pretrain weights in pretrain_folder param.',
                        default='')
    parser.add_argument('--transformer_model',
                        choices=['bert', 'distilbert', 'gpt2', 'roberta', 'transformerxl', 'xlnet', 'albert'],
                        help='Name of the transformer model.',
                        default='roberta')
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization.',
                        default=1)

    args = parser.parse_args()
    main(args)
