# Now we will set up the environment and import libraries.
# One issue with Transformers is the excessive logging.
# Let's set the verbosity to error only.
# This gives us a clean terminal output.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging as LoggingLibrary
from transformers import logging as TransformersLogging
TransformersLogging.set_verbosity_error()

import streamlit as StLibrary
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

# Now we will define a custom stopping criteria for the model.
# One issue with language models is they generate unwanted artifacts.
# We solve this by creating a class that halts generation upon detecting specific strings.
# This gives us clean and relevant outputs.
class StopOnArtifact(StoppingCriteria):
    def __init__(self, Tokenizer, StopStrings):
        self.StopIds = [Tokenizer.encode(Word, add_special_tokens=False)[-1] for Word in StopStrings]
    def __call__(self, InputIds, Scores, **Kwargs):
        return InputIds[0][-1] in self.StopIds

StLibrary.set_page_config(page_title="PokédexLLM", page_icon="🔥⚡💧🍃", layout="wide")
StLibrary.title("🔥⚡💧🍃 PokédexLLM (Llama 3.1 8B + RAG)")

# Now we will load the language model and vector database.
# One issue with large models is high memory consumption.
# We solve this by enabling 4-bit quantization during loading.
# This gives us efficient inference on limited hardware.
@StLibrary.cache_resource
def LoadAssets():
    ModelPath = "PokedexLLM/pokedex_llama3_model"
    
    Model, Tokenizer = FastLanguageModel.from_pretrained(
        model_name=ModelPath,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(Model)
    Tokenizer = get_chat_template(Tokenizer, chat_template="llama-3.1")

    EmbeddingsModel = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    VectorDatabase = Chroma(
        persist_directory="PokedexLLM/chroma_db", 
        embedding_function=EmbeddingsModel
    )
    return Model, Tokenizer, VectorDatabase

Model, Tokenizer, VectorDatabase = LoadAssets()
UserPrompt = StLibrary.chat_input("Ask about a Pokémon...")

if UserPrompt:
    with StLibrary.chat_message("user"):
        StLibrary.markdown(UserPrompt)

    with StLibrary.chat_message("assistant"):
        # Now we will retrieve relevant information chunks for the query.
        # One issue with basic prompts is they lack specific context.
        # We solve this by searching the vector database and building a context block.
        # This gives us factually grounded answers.
        RetrievedDocs = VectorDatabase.similarity_search(UserPrompt, k=1)
        
        RefinedContextList = [DocItem.page_content for DocItem in RetrievedDocs]
        ContextInfo = "\n".join(set(RefinedContextList))
        
        SystemInstruction = "You are a specialized digital encyclopedia designed to provide highly accurate, analytical, and competitive data regarding Pokémon."
        
        MessageList = [
            {"role": "system", "content": SystemInstruction},
            {"role": "user", "content": f"Context:\n{ContextInfo}\n\nQuestion: {UserPrompt}"}
        ]
        
        # Now we will format the input for the language model.
        # One issue with raw text is the model expects a specific prompt format.
        # We solve this by applying the tokenizer chat template.
        # This gives us correctly structured inputs for generation.
        FullPrompt = Tokenizer.apply_chat_template(MessageList, tokenize=False, add_generation_prompt=True)
        InputsData = Tokenizer([FullPrompt], return_tensors="pt").to("cuda")
        InputLength = InputsData.input_ids.shape[1]
        
        # Now we will generate the text response.
        # One issue with open-ended generation is the model can run too long.
        # We solve this by applying custom stopping criteria and terminators.
        # This gives us concise answers without trailing artifacts.
        ArtifactStopper = StopOnArtifact(Tokenizer, ["Enough", "thinking", "Question", "Context", "assistant"])
        StopCriteria = StoppingCriteriaList([ArtifactStopper])
        
        MyTerminators = [Tokenizer.eos_token_id, Tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        
        with StLibrary.spinner("Analyzing the Pokédex..."):
            OutputTokens = Model.generate(
                **InputsData, 
                max_new_tokens=512, 
                temperature=0.1, 
                top_p=0.9,
                repetition_penalty=1.1, 
                use_cache=True,
                eos_token_id=MyTerminators,
                pad_token_id=Tokenizer.eos_token_id,
                stopping_criteria=StopCriteria
            )

        # Now we will clean up the generated output.
        # One issue with raw generation is it can contain broken sentences at the end.
        # We solve this by slicing the string at the last punctuation mark.
        # This gives us a polished and readable final response.
        ResponseTokens = OutputTokens[0][InputLength:]
        FinalAnswer = Tokenizer.decode(ResponseTokens, skip_special_tokens=True).strip()
        
        FinalAnswer = FinalAnswer.split("Enough")[0].split("thinking")[0].split("Question")[0].strip()
        
        LastPeriodIndex = FinalAnswer.rfind('.')
        LastExclamationIndex = FinalAnswer.rfind('!')
        LastQuestionIndex = FinalAnswer.rfind('?')

        LastPunctuationIndex = max(LastPeriodIndex, LastExclamationIndex, LastQuestionIndex)

        if LastPunctuationIndex != -1:
            FinalAnswer = FinalAnswer[:LastPunctuationIndex + 1]
        
        StLibrary.markdown(FinalAnswer)