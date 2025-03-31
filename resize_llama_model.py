# prompt: write python script which take the tokenizer path and resize the input embedding based on len of tokenizer for llama3.2 1b model
# python file

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def resize_embedding(tokenizer_path, model_path="meta-llama/Llama-3.2-1B"):
    """
    Resizes the input embedding of a Llama 3.2 model based on the tokenizer's vocabulary size.

    Args:
        tokenizer_path (str): Path to the tokenizer.
        model_path (str): Path to the pre-trained Llama 3.2 model. Defaults to "meta-llama/Llama-3.2-1B".
    """
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", use_safetensors=True, torch_dtype=torch.bfloat16
        )

        # Resize the token embeddings
        model.resize_token_embeddings(len(tokenizer))
        print(f"Successfully resized embedding to match tokenizer size: {len(tokenizer)}")
        
        # Optionally save the updated model
        # model.save_pretrained("./resized_llama_model")
        # print("Resized model saved to ./resized_llama_model")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    tokenizer_path = "output_tokenizer_3_2_1B/"
    resize_embedding(tokenizer_path)