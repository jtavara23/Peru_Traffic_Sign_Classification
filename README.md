# SignalsWindows
Thesis' Project

---------------------------------
For VirtualEnv

sudo apt install python-pip
pip install virtualenvwrapper
source `which virtualenvwrapper.sh`
mkvirtualenv -p python3 env_name
workon env_name
pip install -r reqForThesis.txt


[optional]
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
sudo apt-get install python3-tk

List envs: du -hs /home/hostname/.virtualenvs/


---------------------------------
For Anaconda env
conda remove -n envName --all
conda info --envs
conda create -n tensor35 python=3.5 anaconda
conda activate tensor 35
conda deactivate