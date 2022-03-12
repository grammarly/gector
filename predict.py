import argparse
import os
import time
import datetime
from utils.helpers import read_lines
from gector.gec_model import GecBERTModel
import torch


from difflib import SequenceMatcher

def generate_text_for_log(processed_lines, total_lines, corrected_lines, prediction_duration, cnt_corrections):
    return "Processed lines: "+str(processed_lines)+"/"+str(total_lines)+" = "+ str(round(100*processed_lines/total_lines, 2))+"%\n"+ "Corrected lines: "+ str(corrected_lines)+"/"+str(processed_lines)+" = "+ str(round(100*corrected_lines/processed_lines, 2))+"%\n"+ "Prediction duration: "+ str(prediction_duration)+"\n"+ "Total corrections: "+str(cnt_corrections)


def check_corrected_line(source_tokens, target_tokens):
    matcher = SequenceMatcher(None, source_tokens, target_tokens)
    raw_diffs = list(matcher.get_opcodes())
    if len(raw_diffs) == 1:
        if raw_diffs[0][0] == 'equal':
            return 0
    return 1    
    
def get_corrected_lines_for_batch(source_batch, target_batch):
    corrected = []
    for source, target in zip(source_batch, target_batch):
        corrected.append(check_corrected_line(source, target))
    return corrected
                            
def predict_for_file(input_file, output_file, model, batch_size=32, save_logs=0):
    test_data = read_lines(input_file)
#     predictions = []
    cnt_corrections = 0
    batch = []
    with open(output_file, 'w') as f:
        f.write("")
    
    if save_logs:
        with open(output_file+".log", 'w') as f:
            f.write("")

        with open(output_file+".check_correction", 'w') as f:
            f.write("")
    
    predicting_start_time = time.time()
    
    total_lines = len(test_data)
    processed_lines = 0
    corrected_lines = 0
    
    for sent in test_data:
        batch.append(sent.split())
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            
            processed_lines += batch_size
              
            pred_sents = [" ".join(x) for x in preds]
            
            with open(output_file, 'a') as f:
                f.write("\n".join(pred_sents) + '\n')
                
            cnt_corrections += cnt
            
            if save_logs:
                checked_lines = get_corrected_lines_for_batch(batch, preds)
                corrected_lines += sum(checked_lines)
                checked_lines = [str(s) for s in checked_lines]
                with open(output_file+".check_correction", 'a') as f:
                    f.write("\n".join(checked_lines) + '\n')
            
                predicting_elapsed_time = time.time() - predicting_start_time
                prediction_duration = datetime.timedelta(seconds=predicting_elapsed_time)

                with open(output_file+".log", 'w') as f:
                    f.write(generate_text_for_log(processed_lines, total_lines, corrected_lines, prediction_duration, cnt_corrections))


            batch = []
    if batch:
        preds, cnt = model.handle_batch(batch)
        processed_lines += len(batch)
        pred_sents = [" ".join(x) for x in preds]   
        
        with open(output_file, 'a') as f:
            f.write("\n".join(pred_sents) + '\n')
        
        cnt_corrections += cnt
        
        checked_lines = get_corrected_lines_for_batch(batch, preds)    
        corrected_lines += sum(checked_lines)
        checked_lines = [str(s) for s in checked_lines]

        if save_logs:
        
            with open(output_file+".check_correction", 'a') as f:
                    f.write("\n".join(checked_lines) + '\n')


            predicting_elapsed_time = time.time() - predicting_start_time
            prediction_duration = datetime.timedelta(seconds=predicting_elapsed_time)

            with open(output_file+".log", 'w') as f:
                f.write(generate_text_for_log(processed_lines, total_lines, corrected_lines, prediction_duration, cnt_corrections))
    
    predicting_elapsed_time = time.time() - predicting_start_time
    prediction_duration = datetime.timedelta(seconds=predicting_elapsed_time)
    
    print(prediction_duration)
    
    return cnt_corrections


def main(args):
    # get all paths
#     if args.count_thread != -1:
#         torch.set_num_threads = str(args.count_thread)
#         os.environ["OMP_NUM_THREADS"] = str(args.count_thread)
#         os.environ["MKL_NUM_THREADS"] = str(args.count_thread)
    
    if args.cuda_device_index != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device_index)
        os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    
    model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=args.model_path,
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         min_error_probability=args.min_error_probability,
                         min_probability=args.min_error_probability,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=args.transformer_model,
                         special_tokens_fix=args.special_tokens_fix,
                         log=False,
                         confidence=args.additional_confidence,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights,
                         use_cpu=bool(args.use_cpu))

    cnt_corrections = predict_for_file(args.input_file, args.output_file, model,
                                       batch_size=args.batch_size, save_logs=args.save_logs)
    # evaluate with m2 or ERRANT
    print(f"Produced overall corrections: {cnt_corrections}")


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Path to the model file.', nargs='+',
                        required=True)
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        required=True)
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        required=True)
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert', 'roberta-large', 'xlnet-large', 'deberta', 'deberta-large', 'bart', 'bart-large', 'bert-large', 't5-base', 'funnel-transformer-medium-base', 'roberta-openai', 'deberta-xx-large', 'deberta-xlarge', 'ukr-roberta-base'],
                        help='Name of the transformer model.',
                        default='roberta')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--min_probability',
                        type=float,
                        default=0.0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        default=0.0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=1)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None),
    parser.add_argument('--cuda_device_index',
                        type=int,
                        help='What card of gpu to use, if -1 use all',
                        default=-1)
    parser.add_argument('--use_cpu',
                        type=int,
                        help='use only cpu',
                        default=0)
    parser.add_argument('--count_thread',
                        type=int,
                        help='count of cpus/threads',
                        default=-1)
    parser.add_argument('--save_logs',
                        type=int,
                        help='count of cpus/threads',
                        default=0)
    args = parser.parse_args()
    main(args)
