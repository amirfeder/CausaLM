#!/bin/bash
echo "Starting to set up CausaLM environment in 5 seconds..."
sleep 5
echo "Conda Environments Setup"
mkdir ~/bin
cd ~/bin
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh
source ~/anaconda3/etc/profile.d/conda.sh
# mkdir -p ~/dev
# cd ~/dev/
# git clone https://github.com/NVIDIA/apex
# cd apex
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
echo "Clone CausaLM git repository"
mkdir -p ~/dev
cd ~/dev/
git clone https://github.com/amirfeder/CausaLM.git
echo "export CAUSALM_REPO=\$HOME/dev/CausaLM/" >> ~/.bash_profile
# conda env create --file ~/dev/CausaLM/causalm_gpu_env.yml
conda create --name causalm
conda activate causalm
conda install pip
pip install --upgrade numpy scipy matplotlib pandas seaborn scikit-learn gensim nltk pyLDAvis ray jupyter mkl pytest torch torchvision torchtext tensorflow-gpu spacy[cuda] tensorboard streamlit tabulate tqdm statsmodels transformers ignite spacy-transformers jupyter_contrib_nbextensions jupyterlab
pip install git+https://github.com/nadavo/Timer.git
# pip install https://github.com/explosion/spacy-models/releases/download/en_pytt_xlnetbasecased_lg-2.1.1/en_pytt_xlnetbasecased_lg-2.1.1.tar.gz https://github.com/explosion/spacy-models/releases/download/en_pytt_bertbaseuncased_lg-2.1.1/en_pytt_bertbaseuncased_lg-2.1.1.tar.gz https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.1.0/en_core_web_lg-2.1.0.tar.gz
conda deactivate
echo "Google Drive Setup"
cd ~/bin
wget https://dl.google.com/go/go1.13.3.linux-amd64.tar.gz
tar -xzvf go1.13.3.linux-amd64.tar.gz
mkdir -p $HOME/bin/go/packages
echo "export GOROOT=\$HOME/bin/go" >> ~/.bash_profile
echo "export GOPATH=\$GOROOT/packages" >> ~/.bash_profile
echo "export PATH=\$PATH:\$GOROOT/bin:\$GOPATH/bin" >> ~/.bash_profile
echo "alias 'conda_env'='source \$HOME/anaconda3/etc/profile.d/conda.sh'" >> ~/.bash_profile
echo "alias 'causalm_env'='conda_env && conda activate causalm && export PYTHONPATH=\$CAUSALM_REPO:\$PYTHONPATH && cd \$CAUSALM_REPO'" >> ~/.bash_profile
source ~/.bash_profile
go get -u -v github.com/odeke-em/drive/cmd/drive
drive init ~/GoogleDrive
mkdir -p ~/GoogleDrive/AmirNadav/CausaLM/
echo "export CAUSALM_DATA=\$HOME/GoogleDrive/AmirNadav/CausaLM/" >> ~/.bash_profile
# echo "export MOOD_RESULTS=\$NBA_DATA/Results/Play-By-Play/" >> ~/.bash_profile
# echo "export RESOURCES_ID='1tcyR7J0_bvnTjCjkBf3_L-Ay5RQ8N4nS'" >> ~/.bash_profile
# echo "export RESULTS_ID='1MD85gp8wMkFLe2Xhe_5GlofQbe1xY4FM'" >> ~/.bash_profile
# echo "export DATASETS_ID='1xKIUzRmZkNRCX7LElNgl8gx4MmigpXxX'" >> ~/.bash_profile
# echo "alias 'nba_drive_full_sync'='cd \$NBA_DATA && drive pull -verbose -id \$DATASETS_ID && drive pull -verbose -id \$RESOURCES_ID && drive pull -verbose -id \$RESULTS_ID'" >> ~/.bash_profile
# echo "alias 'nba_drive_push'='cd \$NBA_DATA && drive push -verbose -files -fix-clashes'" >> ~/.bash_profile
# echo "alias 'nba_drive_pull'='cd \$NBA_DATA && drive pull -verbose -id \$1'" >> ~/.bash_profile
source ~/.bash_profile
# cd ~/GoogleDrive/AmirNadav/CausaLM/
# drive pull -verbose -no-prompt -id 1tcyR7J0_bvnTjCjkBf3_L-Ay5RQ8N4nS # sync Resources from Google Drive
# drive pull -verbose -no-prompt -id 1MD85gp8wMkFLe2Xhe_5GlofQbe1xY4FM # sync Results from Google Drive
# drive pull -verbose -no-prompt -id 1xKIUzRmZkNRCX7LElNgl8gx4MmigpXxX # sync Datasets from Google Drive
echo "Finished setting up CausaLM environment!"
