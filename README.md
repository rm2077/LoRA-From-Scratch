# LoRA From Scratch

Implemented from the paper "Low-Rank Adaptation of Language Models" in order to improve efficiency of fine-tuning. 
I used my implementation on a BERT Sequence Classification model for a sentiment analysis task (IMDB reviews). 

# How does LoRA work?

We freeze all of the parameters of our language model and add two multiplied learnable matrices A and B to selected frozen weights.
LoRA is usually applied to the Attention mechanism of a language model, specifically the query, key, and value matrices. 

By doing this, we only fine tune a small percentage of the parameters of a language model rather than fine-tuning
all of the LLM's parameters, increasing efficiency and potentially results.




