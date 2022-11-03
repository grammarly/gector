import filecmp
from pathlib import Path
import requests
import tempfile
from tqdm import tqdm

from gector.gec_model import GecBERTModel
from gector.utils.helpers import read_lines

ORIG_FILE_DIR = Path(__file__).parent / "original"
GOLD_FILE_DIR = Path(__file__).parent / "prediction"
TEST_FIXTURES_DIR_PATH = Path(__file__).parent.parent / "test_fixtures"
VOCAB_PATH = TEST_FIXTURES_DIR_PATH.joinpath("roberta_model/vocabulary")
MODEL_URL = "https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th"


def download_weights():
    """
    Downloads model weights from S3 if not already present at path.

    Returns
    -------
    Path
        Path to model weights file
    """

    model_path = TEST_FIXTURES_DIR_PATH / "roberta_model" / "weights.th"
    if not model_path.exists():
        response = requests.get(MODEL_URL)
        with model_path.open("wb") as out_fp:
            # Write out data with progress bar
            for data in tqdm(response.iter_content()):
                out_fp.write(data)
    assert model_path.exists()

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
        batch.append(sent.split())
        if len(batch) == batch_size:
            preds, cnt = model.handle_batch(batch)
            predictions.extend(preds)
            batch = []
    if batch:
        preds, cnt = model.handle_batch(batch)
        predictions.extend(preds)

    result_lines = [" ".join(pred) for pred in predictions]

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
    Generate predictions for all test files and test that there are no changes.

    Parameters
    ----------
    model : GecBERTModel
        Initialized model
    """

    for child in ORIG_FILE_DIR.iterdir():
        if child.is_file():
            input_file = str(child.resolve())
            gold_standard_file = str(GOLD_FILE_DIR.joinpath(child.name))
            # Create temp file to store generated output
            with tempfile.NamedTemporaryFile() as temp_file:
                predict_for_file(input_file, temp_file, model)
                compare_files(child.name, gold_standard_file, temp_file.name)


def main():

    # Download weights from S3
    model_path = download_weights()

    # Initialize model
    model = GecBERTModel(
        vocab_path=VOCAB_PATH,
        model_paths=[model_path],
        max_len=50,
        min_len=3,
        iterations=5,
        min_error_probability=0.0,
        lowercase_tokens=0,
        model_name="roberta",
        special_tokens_fix=1,
        log=False,
        confidence=0,
        del_confidence=0,
        is_ensemble=0,
        weights=None,
    )

    # Generate predictions and compare to previous output.
    predict_and_compare(model)


if __name__ == "__main__":
    main()
