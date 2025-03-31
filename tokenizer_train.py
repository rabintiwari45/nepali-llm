import argparse
import sentencepiece as spm

def train_tokenizer(data_file, model_prefix, vocab_size, num_threads, model_type, max_sentence_length, shuffle_input_sentence, character_coverage, hard_vocab_limit):
    spm.SentencePieceTrainer.train(
        input=data_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        num_threads=num_threads,
        model_type=model_type,
        max_sentence_length=max_sentence_length,
        shuffle_input_sentence=shuffle_input_sentence,
        character_coverage=character_coverage,
        hard_vocab_limit=hard_vocab_limit,
    )

def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece Tokenizer")
    parser.add_argument('--data_file', type=str, required=True, help='Input data file')
    parser.add_argument('--model_prefix', type=str, required=True, help='Model prefix for the tokenizer')
    parser.add_argument('--vocab_size', type=int, default=16000, help='Vocabulary size')
    parser.add_argument('--num_threads', type=int, default=12, help='Number of threads')
    parser.add_argument('--model_type', type=str, default='bpe', help='Model type (e.g., bpe, unigram, char, word)')
    parser.add_argument('--max_sentence_length', type=int, default=1073741824, help='Maximum sentence length')
    parser.add_argument('--shuffle_input_sentence', type=str, default='true', help='Shuffle input sentences')
    parser.add_argument('--character_coverage', type=float, default=1.0, help='Character coverage')
    parser.add_argument('--hard_vocab_limit', type=str, default='false', help='Hard vocabulary limit')

    args = parser.parse_args()

    train_tokenizer(
        data_file=args.data_file,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        num_threads=args.num_threads,
        model_type=args.model_type,
        max_sentence_length=args.max_sentence_length,
        shuffle_input_sentence=args.shuffle_input_sentence,
        character_coverage=args.character_coverage,
        hard_vocab_limit=args.hard_vocab_limit,
    )

if __name__ == "__main__":
    main()