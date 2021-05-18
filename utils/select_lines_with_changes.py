import argparse
import os
import time
import datetime

from difflib import SequenceMatcher

import spacy
nlp = spacy.load("en_core_web_sm", exclude=['tagger', 'parser', 'ner', 'lemmatizer', 'textcat', 'custom'])

import numpy as np
from tqdm import tqdm

def read_lines(fn):
    if not os.path.exists(fn):
        return []
    with open(fn, 'r', encoding='utf-8') as f:
        text = f.read()
    lines = text.split("\n")
    return lines

def write_lines(fn, lines, mode='w'):
    text_to_write = "\n".join(lines)
    if len(text_to_write) > 0:
        text_to_write + "\n"
    with open(fn, encoding='utf-8', mode=mode) as f:
        f.write(text_to_write)

def check_corrected_line(source_tokens, target_tokens):
    matcher = SequenceMatcher(None, source_tokens, target_tokens)
    raw_diffs = list(matcher.get_opcodes())
    if len(raw_diffs) == 1:
        if raw_diffs[0][0] == 'equal':
            return 0
    return 1

def compare_lines(source, target):
    source_tokens = [token.text for token in list(nlp(source))]
    target_tokens = [token.text for token in list(nlp(target))]
    return check_corrected_line(source_tokens, target_tokens)
 

def generate_text_for_log(processed_lines, 
                          total_lines, 
                          corrected_lines, 
                          prediction_duration):
    return "Processed lines: "+str(processed_lines)+"/"+str(total_lines)+" = "+ str(round(100*processed_lines/total_lines, 2))+"%\n"+ "Corrected lines: "+ str(corrected_lines)+"/"+str(processed_lines)+" = "+ str(round(100*corrected_lines/processed_lines, 2))+"%\n"+ "Check duration: "+ str(prediction_duration)+"\n"    
    
def select_lines_with_changes(input_source, input_target, 
                              output_source, output_target,
                              output_log, chunk_size,
                              start_line, stop_line,
                              count_cpu):

    write_lines(output_log, ["Start"], mode='w')
    
    input_source_lines = read_lines(input_source)
    input_target_lines = read_lines(input_target)
    
    output_source_lines = []
    output_target_lines = []
    
    max_stop_line = min(len(input_source_lines), len(input_target_lines))
    write_lines(output_log, ["min_stop_line = "+str(max_stop_line)], mode='w')
    
    if stop_line > 0 and stop_line <= max_stop_line:
        input_source_lines = input_source_lines[start_line : stop_line]
        input_target_lines = input_target_lines[start_line : stop_line]
    else:
        input_source_lines = input_source_lines[start_line : max_stop_line]
        input_target_lines = input_target_lines[start_line : max_stop_line]
        
    
    assert len(input_source_lines) == len(input_target_lines), "Different count of lines"
    
    total_lines = len(input_source_lines)
    
    predicting_start_time = time.time()

    processed_lines = 0
    corrected_lines = 0
    
    for source, target in zip(input_source_lines, input_target_lines):
        if compare_lines(source, target) == 1:
            output_source_lines.append(source)
            output_target_lines.append(target)
            corrected_lines += 1
        
        processed_lines += 1
        if processed_lines % chunk_size == 0:
            predicting_elapsed_time = time.time() - predicting_start_time
            prediction_duration = datetime.timedelta(seconds=predicting_elapsed_time)
            text_for_log = generate_text_for_log(processed_lines, 
                          total_lines, 
                          corrected_lines, 
                          prediction_duration)
            
            write_lines(output_log, [text_for_log], mode='w')
            write_lines(output_source, output_source_lines, mode='a')
            write_lines(output_target, output_target_lines, mode='a')
            output_source_lines = []
            output_target_lines = []
            
    write_lines(output_source, output_source_lines, mode='a')
    write_lines(output_target, output_target_lines, mode='a')
    text_for_log = generate_text_for_log(processed_lines, 
                          total_lines, 
                          corrected_lines, 
                          prediction_duration)
            
    write_lines(output_log, [text_for_log], mode='w')
    
    

def main(args):
    select_lines_with_changes(args.input_source, args.input_target, 
                              args.output_source, args.output_target, 
                              args.output_log, args.chunk_size, args.start_line, 
                              args.stop_line, args.count_cpu)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-is', '--input_source',
                        help='Path to the input source file',
                        required=True)
    parser.add_argument('-it', '--input_target',
                        help='Path to the input target file',
                        required=True)
    parser.add_argument('-os', '--output_source',
                        help='Path to the output source file',
                        required=True)
    parser.add_argument('-ot', '--output_target',
                        help='Path to the output tatget file',
                        required=True)
    parser.add_argument('-ol', '--output_log',
                        help='Path to the output log file',
                        required=True)
    parser.add_argument('--chunk_size',
                        type=int,
                        help='Dump each chunk size.',
                        default=1000)
    parser.add_argument('--start_line',
                        type=int,
                        help='From which line to start',
                        default=0)
    parser.add_argument('--stop_line',
                        type=int,
                        help='On which line to end',
                        default=-1)
    parser.add_argument('--count_cpu',
                        type=int,
                        help='how many cpu to use',
                        default=1)
    
    args = parser.parse_args()
    main(args)
