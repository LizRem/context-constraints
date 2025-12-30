import pandas as pd
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, processors, decoders, Regex

data = pd.read_csv("pretraining_data.csv")

# Get unique terms for tokenizer
unique_strings = data['term'].unique().tolist()
print(f"Creating {len(unique_strings)} custom tokens")

def get_frequent_terms(data, min_frequency=3, max_vocab_size=50000):
    term_counts = data['term'].value_counts()
    frequent_terms = term_counts[term_counts >= min_frequency].index.tolist()
    
    if len(frequent_terms) > max_vocab_size:
        frequent_terms = frequent_terms[:max_vocab_size]
    
    print(f"Filtered to {len(frequent_terms)} terms (min_freq={min_frequency})")
    return frequent_terms

# Get filtered vocabulary
filtered_terms = get_frequent_terms(data, min_frequency=3, max_vocab_size=50000)

def create_llama_medical_tokenizer(unique_terms, max_length=131072):
    """
    Create tokenizer matching actual LLaMA 3.1 special tokens
    """
    # Use LLaMA 3.1's actual special tokens
    vocab = {
        '<|begin_of_text|>': 0,     # LLaMA's BOS token
        '<|end_of_text|>': 1,       # LLaMA's EOS token (also used for padding)
        '<unk>': 2,                 # Add our own UNK token for medical terms
    }
    
    # Add medical terms starting from ID 3
    for i, term in enumerate(unique_terms, start=3):
        vocab[term] = i
    
    print(f"Total vocabulary size: {len(vocab)} tokens")
    
    # Create tokenizer
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token='<unk>'))
    tokenizer.normalizer = None 
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=Regex(r" "),
        behavior="removed"
    )
    
    # Use LLaMA's actual special tokens for processing
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|begin_of_text|> $A <|end_of_text|>",
        special_tokens=[
            ("<|begin_of_text|>", vocab["<|begin_of_text|>"]),
            ("<|end_of_text|>", vocab["<|end_of_text|>"]),
        ],
    )
    
    tokenizer.decoder = decoders.Sequence([
        decoders.Replace("<|end_of_text|>", ""),
        decoders.Replace("<|begin_of_text|>", ""),
        decoders.Strip()
    ])
    
    # Create HuggingFace tokenizer with LLaMA's exact settings # to ensure we can use it with out LLaMA models
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token='<unk>',
        bos_token='<|begin_of_text|>',     
        eos_token='<|end_of_text|>',      
        pad_token='<|end_of_text|>',      
        model_max_length=max_length,
        padding_side='left',              
        truncation_side='left'
    )
    
    return hf_tokenizer, vocab

# Create tokenizer
llama_medical_tokenizer, vocab = create_llama_medical_tokenizer(filtered_terms, max_length=131072)
  
# Verify it worked
print(f"Tokenizer max length: {llama_medical_tokenizer.model_max_length}")
print(f"Actual vocabulary size: {len(llama_medical_tokenizer.get_vocab())}")
print(f"Special tokens: {llama_medical_tokenizer.special_tokens_map}")
print(f"Padding side: {llama_medical_tokenizer.padding_side}")

# Check what number the special tokens are assigned
print(f"\nBOS token: '{llama_medical_tokenizer.bos_token}' (ID: {llama_medical_tokenizer.bos_token_id})")
print(f"EOS token: '{llama_medical_tokenizer.eos_token}' (ID: {llama_medical_tokenizer.eos_token_id})")

llama_medical_tokenizer.save_pretrained("tokenizer_llama")
# this tokenizer expects strings of medical codes, separated by a space
