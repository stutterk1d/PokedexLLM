# PokedexLLM

This repository is a data science project for a chatbot specializing in Pokemon. You can find the code to retrieve factual data and generate answers about Pokémon here. I built an artificial intelligence pipeline using Llama 3.1 8B and a Chroma vector database to analyze different creatures. I also applied PEFT and LoRA to train the model efficiently. You can also run a Streamlit application to interact with the system.  Below is an example of the application.

https://github.com/user-attachments/assets/bdf0277a-f0a7-4df3-913c-0914fcde92b1

## Project Structure


## PEFT and LoRA Implementation

I used Parameter-Efficient Fine-Tuning (PEFT) to train the model. Updating all the parameters takes too much memory. PEFT solves this problem. 

I applied Low-Rank Adaptation (LoRA) to specific model modules. You can see my configuration below. 

* **Target Modules**: I attached LoRA matrices to the projection components. 
* **LoRA Rank**: I set the rank to 32. 
* **LoRA Alpha**: I used an alpha of 64. 
* **Gradient Checkpointing**: I activated the Unsloth checkpointing feature. 

## Technologies Used

You will mainly use these tools for this project:

1. Python
2. Unsloth
3. BAAI/bge-large-en-v1.5
4. Chroma
5. Llama 3.1 8B
6. Streamlit

## How to Run the Code


## Data Details

I configured the system to manage hardware resources.

| Component | Purpose |
| :--- | :--- |
| Llama 3.1 8B | Semantic reasoning |
| Chroma | Factual retrieval |

The vector database process gives a strict limit of one document per retrieval. The 4-bit quantization handles high memory consumption and stabilizes the system for consumer hardware.

## Future Work

The current pipeline only explores a fraction of what is possible. You can expand this project by doing things like:

* Testing other open-source models like Mistral.
* Adding an image recognition feature for Pokémon sprites.
