conda create -n 3UTRBERT
conda activate 3UTRBERT

conda install install python=3.8 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
git clone https://github.com/yangyn533/3UTRBERT
cd 3UTRBERT
python3 -m pip install --editable .
python3 -m pip install -r requirements.txt


##only run if not all the packages installed properly above
#pip install seaborn
#pip install transformers
#pip install pyfaidx
#pip install python-decouple
#pip install sacremoses
#pip install boto3
#pip install sentencepiece
#pip install Bio
#pip install pyahocorasick
