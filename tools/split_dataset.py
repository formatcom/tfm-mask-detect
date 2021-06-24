import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


# Configuration
PATH_OI_DATASET = '{}/images'.format(os.getcwd())
PATH_AI_DATASET = '{}/annotations'.format(os.getcwd())

# Clean Enviroment
shutil.rmtree('out', ignore_errors=True)
os.mkdir('out')
os.chdir('out')

os.mkdir('train')
os.mkdir('val')

os.mkdir('a_train')
os.mkdir('a_val')

# Logic
dataset = os.listdir(PATH_OI_DATASET)

train, val = train_test_split(dataset, test_size=0.2, random_state=1)

for item in train:
    _name = Path(item)
    shutil.copy('{}/{}'.format(PATH_OI_DATASET, _name), 'train/{}'.format(_name))
    shutil.copy('{}/{}'.format(PATH_AI_DATASET, _name.with_suffix('.xml')),
                                            'a_train/{}'.format(_name.with_suffix('.xml')))

for item in val:
    _name = Path(item)
    shutil.copy('{}/{}'.format(PATH_OI_DATASET, _name), 'val/{}'.format(_name))
    shutil.copy('{}/{}'.format(PATH_AI_DATASET, _name.with_suffix('.xml')),
                                            'a_val/{}'.format(_name.with_suffix('.xml')))
