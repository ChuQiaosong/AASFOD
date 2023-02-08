

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc


from datasets.cityscape import cityscape

from datasets.foggy_cityscape import foggy_cityscape


from datasets.foggy_cityscape_similar import foggy_cityscape_similar
from datasets.foggy_cityscape_disimilar import foggy_cityscape_disimilar


import numpy as np


###########################################
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval','trainval_cg']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))



for year in ['2007']:
  for split in ['train', 'val', 'train_combine_fg', 'train_cg_fg']:
    name = 'cs_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split: cityscape(split, year))

for year in ['2007']:
  for split in ['train', 'val', 'train_combine','train_cg']:
    name = 'cs_fg_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split: foggy_cityscape(split, year))





for year in ['2007']:
  for split in ['train', 'trained']:
    name = 'cs_fg_similar_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split: foggy_cityscape_similar(split, year))

for year in ['2007']:
  for split in ['train', 'trained']:
    name = 'cs_fg_disimilar_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split: foggy_cityscape_disimilar(split, year))




def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
