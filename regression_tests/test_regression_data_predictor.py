import filecmp
from pathlib import Path
import requests
import tempfile
import torch
from tqdm import tqdm

from allennlp.predictors import Predictor
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import Token
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField
from allennlp.data import Batch

from gector.gec_predictor import GecPredictor
from gector.basic_field_embedder import BasicTextFieldEmbedder

# These imports are required so that instantiating the predictor can be done automatically
from gector.datareader import Seq2LabelsDatasetReader
from gector.seq2labels_model import Seq2Labels
from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.tokenizer_indexer import PretrainedBertIndexer
from gector.utils.helpers import read_lines

ORIG_FILE_DIR = Path(__file__).parent / "original"
GOLD_FILE_DIR = Path(__file__).parent / "prediction"
TEST_FIXTURES_DIR_PATH = Path(__file__).parent.parent / "test_fixtures"


def download_weights():
    """
    Downloads model weights from S3 if not already present at path.

    Returns
    -------
    Path
        Path to model directory
    """

    # Download weights for model archive
    weights_url = "https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th"
    model_path = TEST_FIXTURES_DIR_PATH / "roberta_model" / "weights.th"
    if not model_path.exists():
        response = requests.get(weights_url)
        with model_path.open("wb") as out_fp:
            # Write out data with progress bar
            for data in tqdm(response.iter_content()):
                out_fp.write(data)
    model_path = TEST_FIXTURES_DIR_PATH / "roberta_model"

    return model_path


def predict_for_file(input_file, temp_file, model, batch_size=32):
    """
    Generates predictions for a single file and store it in a temp file.

    Parameters
    ----------
    input_file : str
        Path to input file
    temp_file : TemporaryFileWrapper
        Temp file object
    model : GecBERTModel
        Initialized model object
    batch_size : int, optional
        Batch size, by default 32

    Returns
    -------
    int
        Total number of corrections made
    """

    test_data = read_lines(input_file)
    predictions = []
    batch = []
    for sent in test_data:
        batch.append(sent)
        if len(batch) == batch_size:
            preds = model.predict_batch(batch)
            preds_corrected_words = [x["corrected_words"] for x in preds]
            predictions.extend(preds_corrected_words)
            batch = []
    if batch:
        preds = model.predict_batch(batch)
        preds_corrected_words = [x["corrected_words"] for x in preds]
        predictions.extend(preds_corrected_words)

    result_lines = [" ".join(x) for x in predictions]

    with open(temp_file.name, "w") as f:
        f.write("\n".join(result_lines) + "\n")


def compare_files(filename, gold_file, temp_file):
    """
    Compares two files and tests that they are equal.

    Parameters
    ----------
    filename : str
        Name of file being compared
    gold_file : str
        Path to gold standard file
    temp_file : str
        Path to file containing generated prediction
    """

    assert filecmp.cmp(
        gold_file, temp_file, shallow=False
    ), f"Output of {filename} does not match gold output."
    print(filename, "passed.")


def predict_and_compare(model):
    """
    Generate predictions for all test files and tests that there are no changes.

    Parameters
    ----------
    model : Predictor
        Initialized model
    """

    for child in ORIG_FILE_DIR.iterdir():
        if child.is_file():
            input_file = str(ORIG_FILE_DIR.joinpath(child.name))
            gold_standard_file = str(GOLD_FILE_DIR.joinpath(child.name))
            # Create temp file to store generated output
            with tempfile.NamedTemporaryFile() as temp_file:
                predict_for_file(input_file, temp_file, model)
                compare_files(child.name, gold_standard_file, temp_file.name)


def main():

    # Download weights from S3
    model_path = download_weights()

    # Initialize model
    overrides = {"model.text_field_embedder.token_embedders.bert.load_weights": False}
    model = Predictor.from_path(model_path, predictor_name="gec-predictor", overrides=overrides)

    # Generate predictions and compare to previous output.
    predict_and_compare(model)


if __name__ == "__main__":
    main()
