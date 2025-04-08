import argparse
import json
from datasets import load_dataset

def convert_dataset_to_txt(dataset, output_file, key='text'):
    with open(output_file, 'a', encoding='utf-8') as out_file:
        for record in dataset:
            out_file.write(f"{record[key].strip()}\n")

def main():
    parser = argparse.ArgumentParser(description="Convert dataset to TXT")
    parser.add_argument('--output_file', type=str, required=True, help='Output TXT file')
    parser.add_argument('--key', type=str, default='text', help='Key to extract text from dataset')

    args = parser.parse_args()

    # Load a dataset from Hugging Face Hub
    dataset = load_dataset("Sakonii/nepalitext-language-model-dataset")["test"]

    convert_dataset_to_txt(dataset, args.output_file, args.key)

if __name__ == "__main__":
    main()