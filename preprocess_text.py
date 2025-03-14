import os
import argparse

from tqdm import tqdm

import sys
sys.path.append("bert_vits2")
from bert_vits2.text.cleaner import clean_text


def get_args():
    parser = argparse.ArgumentParser(description="The text preprocess of g2p.")

    parser.add_argument('--file-path', type=str, default="filelists/libritts_train_text.txt", 
                        help='the file list of fine-tuning dataset')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    txt_path = args.file_path
    with open(txt_path, "r") as f:
        lines = f.readlines()

    output_path = "filelists/tmp.cleaned"
    with open(output_path, "w") as f:
        for line in tqdm(lines):
            utt, spk, text = line.strip().split("|")
            language = "EN"
            norm_text, phones, tones, word2ph = clean_text(
                text, language
            )
            f.write(
                "{}|{}|{}|{}|{}|{}|{}\n".format(
                    utt,
                    spk,
                    language,
                    norm_text,
                    " ".join(phones),
                    " ".join([str(i) for i in tones]),
                    " ".join([str(i) for i in word2ph]),
                )
            )


if __name__ == "__main__":
    main()