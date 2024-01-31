# whats-the-tab

Developing
cd to the project directory and run:
```
jupyter notebook
```
Then copy the server url, etc. http://localhost:8888/?token=ed084ae
In the ipython notebook python kernel, paste the url into the open by url option.


### Installation
Set up new conda environment and activate it (dont specify python version, we wil let subsequent command determine that)
    -lol ok right now the env is called directml but that's literally what I'm avoiding to get this cuda torch setup properly

Install stable linux condas python latest cuda version
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
`conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`

conda install conda-forge::matplotlib
conda install conda-forge::librosa
pip install PyGuitarPro
conda install conda-forge::tqdm
conda install conda-forge::datasets
conda install -c conda-forge ipywidgets     # for tqdm progress bar
