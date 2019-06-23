#!/bin/bash
echo "Starting to set up CausaLM environment in 5 seconds..."
sleep 5
echo "Conda Environments Setup"
mkdir ~/bin
cd ~/bin
wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
source ~/anaconda3/etc/profile.d/conda.sh
echo "Conda Pytorch Env Setup"
conda create --name causalm_pytorch
conda activate causalm_pytorch
conda install -y pip numpy scipy pandas tabulate six requests beautifulsoup4 nltk h5py lxml numexpr scikit-learn matplotlib spacy boto3 ftfy pyyaml
conda install -y pytorch torchvision cudatoolkit -c pytorch
conda install -y cudnn
conda deactivate
echo "Conda Keras Env Setup"
conda create --name causalm_keras
conda activate causalm_keras
conda install -y pip numpy scipy pandas tabulate six requests beautifulsoup4 nltk h5py lxml numexpr scikit-learn matplotlib boto3 spacy
conda install -y tensorflow-gpu keras cudnn
conda deactivate
echo "Conda AllenNLP Env Setup"
conda create --name causalm_allennlp
conda activate causalm_allennlp
conda install -y pip tensorflow-gpu cudnn mkl pandas tabulate tqdm cython seaborn beautifulsoup4 scikit-learn numpy scipy nltk lxml matplotlib
pip install allennlp ray torchvision ignite textacy
conda deactivate
echo "Conda BERT Env Setup"
conda create --name causalm_bert
conda activate causalm_bert
conda install -y pip tensorflow-gpu cudnn mkl pandas tabulate tqdm cython seaborn beautifulsoup4 scikit-learn numpy scipy nltk lxml matplotlib
conda install -y pytorch torchvision ignite cudatoolkit -c pytorch
pip install pytorch-pretrained-bert ray textacy psutil setproctitle
pip install --upgrade botocore
# mkdir -p ~/dev
# cd ~/dev/
# git clone https://github.com/NVIDIA/apex
# cd apex
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
conda deactivate
echo "Clone CausaLM git repository"
mkdir -p ~/dev
cd ~/dev/
git clone https://github.com/amirfeder/CausaLM.git
echo "export CAUSALM_REPO=\$HOME/dev/CausaLM/" >> ~/.bash_profile
echo "Google Drive Setup"
cd ~/bin
wget https://dl.google.com/go/go1.12.4.linux-amd64.tar.gz
tar -xzvf go1.12.4.linux-amd64.tar.gz
mkdir -p $HOME/bin/go/packages
echo "export GOROOT=\$HOME/bin/go" >> ~/.bash_profile
echo "export GOPATH=\$GOROOT/packages" >> ~/.bash_profile
echo "export PATH=\$PATH:\$GOROOT/bin:\$GOPATH/bin" >> ~/.bash_profile
echo "alias 'conda_env'='source \$HOME/anaconda3/etc/profile.d/conda.sh'" >> ~/.bash_profile
echo "alias 'causalm_pytorch'='conda_env && conda activate mood_pytorch && export PYTHONPATH=\$MOOD_REPO:\$PYTHONPATH && cd \$MOOD_REPO'" >> ~/.bash_profile
echo "alias 'causalm_keras'='conda_env && conda activate mood_keras && export PYTHONPATH=\$MOOD_REPO:\$PYTHONPATH && cd \$MOOD_REPO'" >> ~/.bash_profile
echo "alias 'causalm_allennlp'='conda_env && conda activate mood_allennlp && export PYTHONPATH=\$MOOD_REPO:\$PYTHONPATH && cd \$MOOD_REPO'" >> ~/.bash_profile
echo "alias 'causalm_bert'='conda_env && conda activate mood_bert && export PYTHONPATH=\$MOOD_REPO:\$PYTHONPATH && cd \$MOOD_REPO'" >> ~/.bash_profile
source ~/.bash_profile
go get -u -v github.com/odeke-em/drive/cmd/drive
# drive init ~/GoogleDrive
# mkdir -p ~/GoogleDrive/AmirNadav/NBA/
# echo "export NBA_DATA=\$HOME/GoogleDrive/AmirNadav/NBA/" >> ~/.bash_profile
# echo "export MOOD_RESULTS=\$NBA_DATA/Results/Play-By-Play/" >> ~/.bash_profile
# echo "export RESOURCES_ID='1tcyR7J0_bvnTjCjkBf3_L-Ay5RQ8N4nS'" >> ~/.bash_profile
# echo "export RESULTS_ID='1MD85gp8wMkFLe2Xhe_5GlofQbe1xY4FM'" >> ~/.bash_profile
# echo "export DATASETS_ID='1xKIUzRmZkNRCX7LElNgl8gx4MmigpXxX'" >> ~/.bash_profile
# echo "alias 'nba_drive_full_sync'='cd \$NBA_DATA && drive pull -verbose -id \$DATASETS_ID && drive pull -verbose -id \$RESOURCES_ID && drive pull -verbose -id \$RESULTS_ID'" >> ~/.bash_profile
# echo "alias 'nba_drive_push'='cd \$NBA_DATA && drive push -verbose -files -fix-clashes'" >> ~/.bash_profile
# echo "alias 'nba_drive_pull'='cd \$NBA_DATA && drive pull -verbose -id \$1'" >> ~/.bash_profile
# source ~/.bash_profile
# cd ~/GoogleDrive/AmirNadav/NBA/
# drive pull -verbose -no-prompt -id 1tcyR7J0_bvnTjCjkBf3_L-Ay5RQ8N4nS # sync Resources from Google Drive
# drive pull -verbose -no-prompt -id 1MD85gp8wMkFLe2Xhe_5GlofQbe1xY4FM # sync Results from Google Drive
# drive pull -verbose -no-prompt -id 1xKIUzRmZkNRCX7LElNgl8gx4MmigpXxX # sync Datasets from Google Drive
echo "Finished setting up CausaLM environment!"
