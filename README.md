# EECS 595 Project

This is the final project report for Umich EECS 595 Natural Language Processing, where we implemented 2 innovative approaches for improving an existing question answering system based on BERT or BERT-related pre-trained models. 

## Requirements

The codes rely on python v3.7 and pytorch.

The codes are based on [Transformers](https://github.com/huggingface/transformers) v2.3.0. 

You can install the dependencies by cd to the root directory and run `conda install -r requirements`.

## Training

Here is how to train the model. 

### Answerability Verifier (Yin Yuan)

For this part, Yin Yuan fine-tune an albert-based model on his own computer, which is too large to be uploaded. You can either fine-tune the model by scripts in the sample code as well, or download a fine-tuned version from https://huggingface.co/.

To fine-tune the model, please run the `run_squad.sh` in `sample_code` directory.

After you get the fine-tuned model, you can open `/src/Binary_Answerability_Verifier.ipynb` in jupyter notebook and run all blocks in sequential order. The accuracy value would occur in the end of this notebook. 

## 2-Stage Reading with Reduced Range

For this part, Cheng Qian create a 2 stage reading structure. Here, we uploaded the sentences embeddings using the model `deepset/roberta-base-squad2` from https://huggingface.co/. You can run the 2 stage QA system and get a prediction file `pred_with2stage.json` by running `python 2stageQA.py --model_name deepset/roberta-base-squad2 --reduced_range` based on the model with reduced context range.  Using `--whole_range` instead of `--reduced_range` can let stage 2 do prediction without narrowing the context range. You can also change the model name with other models in https://huggingface.co/, but this may cost some time to run. You can then evaluate the prediction by running `python evaluate.py dev-v2.0.json pred_with2stage.json`.
