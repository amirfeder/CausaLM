## POMS Gender/Race Experimental Pipeline
#### Prerequisites
- Create the CausaLM conda environment: `conda env create --file causalm_gpu_env.yml`
- Install the `en_core_web_lg` spaCy model.
- Make sure to set the `CAUSALM_DIR` variable in `constants.py` to point to the path where the CausaLM datasets are located.
#### Stage 2 training
Run the following scripts in sequence:
- `./pretrain/pregenerate_training_data.py --treatment <gender/race>`
- `./pretrain/mlm_finetune_on_pregenerated.py`
- `./pretrain/genderace_finetune_on_pregenerated.py --treatment <gender/race>`

This will save the intervened BERT language model which treats for Gender or Race treatment.

#### Stage 3 training
- `./pipeline/training.py --treatment <gender/race>`

This will train and test all the POMS classifiers for the full experimental pipeline for Gender or Race treatment.
