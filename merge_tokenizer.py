import argparse
import sentencepiece as spm
import regex as re
from tokenizers import Tokenizer, AddedToken
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

additional_tokens = ['<|end_of_turn|>',
 '<|pad|>',
 '<|im_start|>',
 '[INST]',
 '[/INST]',
 '<<SYS>>',
 '<</SYS>>',
 '<|user|>',
 '<|system|>',
 '<|assistant|>',
 '<|begin_of_text|>',
 '<|start_header_id|>',
 '<|end_header_id|>',
 '<|eot_id|>',
 '<|im_end|>',
 '<|reserved_0|>',
 '<|reserved_1|>',
 '<|reserved_2|>',
 '<|reserved_3|>',
 '<|reserved_4|>',
 '<|reserved_5|>',
 '<|reserved_6|>',
 '<|reserved_7|>',
 '<|reserved_8|>',
 '<|reserved_9|>',
 '<|reserved_10|>',
 '<|reserved_11|>',
 '<|reserved_12|>',
 '<|reserved_13|>',
 '<|reserved_14|>',
 '<|reserved_15|>',
 '<|reserved_16|>',
 '<|reserved_17|>',
 '<|reserved_18|>',
 '<|reserved_19|>',
 '<|reserved_20|>',
 '<|reserved_21|>',
 '<|reserved_22|>',
 '<|reserved_23|>',
 '<|reserved_24|>',
 '<|reserved_25|>',
 '<|reserved_26|>',
 '<|reserved_27|>',
 '<|reserved_28|>',
 '<|reserved_29|>',
 '<|reserved_30|>',
 '<|reserved_31|>',
 '<|reserved_32|>']

def get_tokenizer(version, model_id):
    if version == 0:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif version == 1:
        tokenizer = Tokenizer.from_pretrained(model_id)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    elif version == 2:
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
    return tokenizer

def tokenize_text(text, tokenizer):
    tokens_encoded = tokenizer.encode(text)
    tokens_decoded = []
    for token in tokens_encoded:
        tokens_decoded.append(tokenizer.decode(token))
    print('\n--------------\nNumber of tokens:', len(tokens_decoded))
    print(tokens_decoded)
    return tokens_decoded

def contains_english(text):
    return bool(re.search('[a-zA-Z]', text))

def main():
    parser = argparse.ArgumentParser(description="Tokenizer Script")
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-3.2-1B", help='Model ID for the tokenizer')
    parser.add_argument('--version', type=int, default=0, help='Tokenizer version (0: AutoTokenizer, 1: Fast Tokenizer, 2: Raw BPE)')
    parser.add_argument('--spm_model', type=str, required=True, help='Path to SentencePiece model')
    parser.add_argument('--texts', type=str, nargs='+', required=True, help='List of texts to tokenize')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save the tokenizer')

    args = parser.parse_args()

    tokenizer = get_tokenizer(args.version, args.model_id)

    for text in args.texts:
        tokenize_text(text, tokenizer)

    hi_spm = spm.SentencePieceProcessor()
    hi_spm.Load(args.spm_model)

    nepali_vocab = []
    for i in range(hi_spm.GetPieceSize()):
        token = hi_spm.IdToPiece(i)
        if token.startswith('▁'):
            token = token.replace("▁", " ")
        if not contains_english(token):
            nepali_vocab.append(token)

    new_tokens = []
    base_vocab = tokenizer.get_vocab()
    count = 0
    count_neg = 0
    for token in nepali_vocab:
        if token and token not in base_vocab and token.strip():
            new_tokens.append(token)
            count += 1
        else:
            count_neg += 1
    print(f'added tokne: {count}')
    print(f'not added token: {count_neg}')
    print(f'total tokens: {len(nepali_vocab)}')

    tokenizer.add_tokens(new_tokens)
    tokenizer.add_special_tokens({'additional_special_tokens': additional_tokens})


    for text in args.texts:
        tokenize_text(text, tokenizer)

    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()