import argparse
import os
import time
import datetime

from difflib import SequenceMatcher
from multiprocessing import Pool, cpu_count

import spacy
nlp = spacy.load("en_core_web_sm", exclude=['tagger', 'parser', 'ner', 'lemmatizer', 'textcat', 'custom'])
nlp.add_pipe('sentencizer')

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


def generate_text_for_log(processed_lines, 
                          total_lines, 
                          prediction_duration):
    return "Processed lines: "+str(processed_lines)+"/"+str(total_lines)+" = "+ str(round(100*processed_lines/total_lines, 2))+"%\n"+ "Process duration: "+ str(prediction_duration)+"\n"    


def process_chunk(chunk_lines):
    sentences = []
    for line in chunk_lines:
        doc = nlp(str(line))
        sents = [sent for sent in doc.sents]
        new_sents = [" ".join([tok.text for tok in list(sent)]) for sent in sents]
        sentences.extend(new_sents)
    return {"sentences":sentences}
    

def preprocess_lines(input_f, output_f, output_log, chunk_size,
                              start_line, stop_line,
                              count_cpu):

    write_lines(output_log, ["Start"], mode='w')
    write_lines(output_f, [], mode='w')
    
    input_lines = read_lines(input_f)
    
    total_lines = len(input_lines)
    
    count_chunks = int(total_lines/chunk_size)
    if count_chunks*chunk_size < total_lines:
        count_chunks += 1
     
    predicting_start_time = time.time()

    processed_lines = 0
    
    for chunk_num in range(count_chunks):
        start = chunk_num*chunk_size
        end = (chunk_num+1)*chunk_size
        chunk = input_lines[start:end]
        
        output_lines = []
        
        all_result_maps = []
        sub_chunks_input_lines = np.array_split(chunk, count_cpu)
        pool = Pool(count_cpu)
        result_map = pool.map(process_chunk, sub_chunks_input_lines)
        pool.close()
        pool.join()
        all_result_maps.extend(result_map)
        
        for res in all_result_maps:
            output_lines.extend(res["sentences"])
           
        predicting_elapsed_time = time.time() - predicting_start_time
        prediction_duration = datetime.timedelta(seconds=predicting_elapsed_time)
        
        output_lines = [sen for sen in output_lines if len(sen) > 5]
        
        processed_lines += len(output_lines)
        
        text_for_log = generate_text_for_log(processed_lines, 
                      total_lines, prediction_duration)
        
        write_lines(output_log, [text_for_log], mode='w')
        write_lines(output_f, output_lines, mode='a')
        
        
    
def main(args):
    preprocess_lines(args.input_f, args.output_f, args.log, args.chunk_size, args.start_line, 
                              args.stop_line, args.count_cpu)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_f',
                        help='Path to the input source file',
                        required=True)
    parser.add_argument('-o', '--output_f',
                        help='Path to the output source file',
                        required=True)
    parser.add_argument('-l', '--log',
                        help='Path to the output log file',
                        required=True)
    parser.add_argument('--chunk_size',
                        type=int,
                        help='Dump each chunk size.',
                        default=20000)
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
                        default=14)
    
    args = parser.parse_args()
    main(args)
