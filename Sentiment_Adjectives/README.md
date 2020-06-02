## Sentiment Adjectives Experimental Pipeline

### Prerequisites
- Create the CausaLM conda environment: `conda env create --file causalm_gpu_env.yml`
- Install the [`en_core_web_lg`](https://spacy.io/models/en#en_core_web_lg) spaCy model.
- Make sure to set the `CAUSALM_DIR` variable in `constants.py` to point to the path where the CausaLM datasets are located.

### Stage 2 training
Run the following scripts in sequence:
- `./pretrain/pregenerate_training_data.py`
- `./pretrain/mlm_finetune_on_pregenerated.py`
- `./pretrain/ima_finetune_on_pregenerated.py [--control_task]`

This will save the intervened BERT language model which treats for Adjectives treatment (IMA), with the option of adding the PoS tagging control task.

### Stage 3 training and test
- `./pipeline/training.py [--pretrained_control]`

This will train and test all the Sentiment classifiers for the full experimental pipeline for Adjectives treatment, with the option of utilizing the Stage 2 model which employs the PoS tagging control task.
