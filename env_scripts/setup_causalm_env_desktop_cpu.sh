#!/bin/bash
echo "Starting to set up CausaLM environment in 5 seconds..."
sleep 5
if [ ! -d "$HOME/anaconda3" ]
then
    echo "Conda Setup"
    mkdir -p ~/bin
    cd ~/bin
    wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
    bash Anaconda3-2019.10-Linux-x86_64.sh
fi
source ~/anaconda3/etc/profile.d/conda.sh
echo "Conda CausaLM Environment Setup"
conda create --name causalm
conda activate causalm
conda install pip
pip install --upgrade numpy scipy matplotlib pandas seaborn scikit-learn gensim nltk pyLDAvis ray jupyter mkl pytest torch torchvision torchtext tensorflow spacy tensorboard tensorboardx streamlit tabulate tqdm statsmodels transformers ignite jupyter_contrib_nbextensions jupyterlab captum cython
pip install git+https://github.com/nadavo/Timer.git
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.2.5/en_core_web_lg-2.2.5.tar.gz
echo "alias 'conda_env'='source \$HOME/anaconda3/etc/profile.d/conda.sh'" >> ~/.bash_profile
echo "Clone CausaLM git repository"
mkdir -p ~/dev
cd ~/dev
git clone https://github.com/amirfeder/CausaLM.git
echo "export CAUSALM_REPO=\$HOME/dev/CausaLM/" >> ~/.bash_profile
echo "alias 'causalm_env'='conda_env && conda activate causalm && export PYTHONPATH=\$CAUSALM_REPO:\$PYTHONPATH && cd \$CAUSALM_REPO'" >> ~/.bash_profile
echo "export CAUSALM_DATA=\$HOME/GoogleDrive/AmirNadav/CausaLM/" >> ~/.bash_profile
source ~/.bash_profile
echo "Finished setting up CausaLM environment!"
