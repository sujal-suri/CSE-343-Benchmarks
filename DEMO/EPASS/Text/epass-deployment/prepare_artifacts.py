# prepare_artifacts.py
import os
import pickle
import pandas as pd
import numpy as np
import random
from collections import Counter
import requests
import tarfile
import io
import torch # needed for set_seed

print("--- starting artifact preparation ---")

# --- configuration & hyperparameters (match training run) ---
SEED = 42
BASE_WORKING_DIR = "." # save vocab in current directory
DATA_DIR = os.path.join(BASE_WORKING_DIR, "data/dbpedia/")
VOCAB_SAVE_PATH = os.path.join(BASE_WORKING_DIR, "vocab.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

# dbpedia download url
DBPEDIA_URL = "https://s3.amazonaws.com/fast-ai-nlp/dbpedia_csv.tgz"

# define expected csv paths
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# data subset size used for vocab building (!! adjust if different !!)
NUM_TRAIN_SAMPLES = 50000

# vocabulary settings (match training run)
VOCAB_MAX_SIZE = 25000
VOCAB_MIN_FREQ = 3
SPECIALS = ["<unk>", "<pad>"]
NUM_CLASSES = 14 # needed for stratified sampling

# --- utility functions (copied from notebook) ---
def set_seed(seed_value=42):
    # set seed for reproducibility
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"seed set to {seed_value}")

def download_and_extract_dbpedia(url, target_dir):
    # downloads and extracts dbpedia if not present
    train_path = os.path.join(target_dir, "train.csv")
    test_path = os.path.join(target_dir, "test.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"dataset already found in {target_dir}")
        return train_path, test_path

    print(f"downloading dbpedia dataset from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:gz') as tar:
            members_to_extract = []
            expected_files = ['dbpedia_csv/train.csv', 'dbpedia_csv/test.csv',
                              'dbpedia_csv/classes.txt', 'dbpedia_csv/readme.txt']
            for member in tar.getmembers():
                if member.name in expected_files and member.isfile():
                    member.name = os.path.basename(member.name)
                    members_to_extract.append(member)

            if not members_to_extract or 'train.csv' not in [m.name for m in members_to_extract]:
                 raise ValueError("could not find expected files (train.csv, test.csv) in the archive.")

            print(f"extracting files to {target_dir}...")
            tar.extractall(path=target_dir, members=members_to_extract)

        if not (os.path.exists(train_path) and os.path.exists(test_path)):
             raise FileNotFoundError("extraction failed or train.csv/test.csv not found after extraction.")

        print("dataset downloaded and extracted successfully.")
        return train_path, test_path

    except requests.exceptions.RequestException as e:
        print(f"error downloading dataset: {e}")
        raise
    except tarfile.TarError as e:
        print(f"error extracting dataset: {e}")
        raise
    except Exception as e:
        print(f"an unexpected error occurred during download/extraction: {e}")
        raise

def load_data(csv_path, num_samples=None, seed=42):
    # loads data, renames columns, handles labels, and subsamples
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"csv file not found at {csv_path}. please ensure download was successful.")

    try:
        df = pd.read_csv(csv_path, header=None)
        df.columns = ['label', 'title', 'content']
        df['label'] = df['label'] - 1 # map labels 1-14 to 0-13
        df["text"] = df["title"].astype(str) + " " + df["content"].astype(str)
        df = df[['label', 'text']]

        if num_samples is not None and num_samples < len(df):
            print(f"subsampling to {num_samples} samples...")
            try:
                # stratified sampling logic
                min_samples_per_group = 1
                required_total_samples = df['label'].nunique() * min_samples_per_group
                if num_samples >= required_total_samples and df['label'].nunique() == NUM_CLASSES:
                     df_sampled = df.groupby('label', group_keys=False).apply(
                         lambda x: x.sample(n=max(min_samples_per_group, int(np.rint(num_samples * len(x) / len(df)))), random_state=seed),
                         include_groups=False
                     )
                     actual_sampled = len(df_sampled)
                     print(f"  stratified sample size attempted: {actual_sampled}")
                     if actual_sampled < num_samples:
                         print(f"  warning: stratified sampling resulted in {actual_sampled} samples. topping up randomly.")
                         remaining_needed = num_samples - actual_sampled
                         additional_indices = df.index.difference(df_sampled.index)
                         if len(additional_indices) >= remaining_needed:
                            additional_samples = df.loc[additional_indices].sample(n=remaining_needed, random_state=seed)
                            df = pd.concat([df_sampled, additional_samples], ignore_index=True)
                         else:
                             df = pd.concat([df_sampled, df.loc[additional_indices]], ignore_index=True)
                             print(f"  warning: topped up to {len(df)} samples, which is less than the requested {num_samples}.")
                     else:
                         df = df_sampled.sample(n=num_samples, random_state=seed)
                else:
                     print(f"  not enough samples/classes ({df['label'].nunique()}) for stratification or NUM_CLASSES mismatch, using random sampling.")
                     df = df.sample(n=num_samples, random_state=seed)
            except Exception as e:
                print(f"  stratified sampling failed ({e}), falling back to random sampling.")
                df = df.sample(n=num_samples, random_state=seed)
        else:
             print(f"using full dataset ({len(df)} samples) for vocab.")

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"error loading data from {csv_path}: {e}")
        raise

def simple_tokenizer(text):
    return text.lower().split()

def build_vocab(texts, max_size=None, min_freq=2, specials=SPECIALS):
    print(f"building vocabulary (max_size={max_size}, min_freq={min_freq})...")
    counter = Counter()
    total_texts = len(texts)
    for i, text in enumerate(texts):
        if not isinstance(text, str):
             continue
        tokens = simple_tokenizer(text)
        counter.update(tokens)
        if (i + 1) % 10000 == 0:
            print(f'\r  processed {i + 1}/{total_texts} texts for vocab', end='')
    print(f'\r  finished processing {total_texts} texts for vocab.')

    vocab = {token: idx for idx, token in enumerate(specials)}
    vocab_idx = len(specials)
    words_and_freqs = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    added_tokens = 0
    for word, freq in words_and_freqs:
        if max_size and vocab_idx >= max_size:
            break
        if freq < min_freq:
            continue
        if word not in vocab:
            vocab[word] = vocab_idx
            vocab_idx += 1
            added_tokens += 1

    print(f"vocabulary size: {len(vocab)} (added {added_tokens} tokens meeting criteria)")
    return vocab

# --- main execution ---
if __name__ == "__main__":
    set_seed(SEED)

    try:
        # ensure data exists
        print("checking for dbpedia dataset...")
        train_csv_path, _ = download_and_extract_dbpedia(DBPEDIA_URL, DATA_DIR)
        print(f"using train csv: {train_csv_path}")

        # load data
        print(f"\nloading training data (up to {NUM_TRAIN_SAMPLES} samples)...")
        df_train_for_vocab = load_data(train_csv_path, NUM_TRAIN_SAMPLES, seed=SEED)
        print(f"loaded {len(df_train_for_vocab)} samples for vocabulary building.")

        # build vocabulary
        vocab = build_vocab(
            df_train_for_vocab["text"].tolist(),
            max_size=VOCAB_MAX_SIZE,
            min_freq=VOCAB_MIN_FREQ,
            specials=SPECIALS
        )

        # save vocabulary
        print(f"\nsaving vocabulary to {VOCAB_SAVE_PATH}...")
        with open(VOCAB_SAVE_PATH, 'wb') as f:
            pickle.dump(vocab, f)
        print("vocabulary saved successfully.")
        print("--- artifact preparation finished ---")

    except FileNotFoundError as e:
        print(f"\nerror: {e}")
        print("please check data paths and ensure the dataset was downloaded correctly.")
    except Exception as e:
        print(f"\nan error occurred: {e}")