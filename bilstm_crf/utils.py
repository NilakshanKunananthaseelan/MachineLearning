import random
import torch
from typing import List,Tuple




def load_dataset(filepath: str)-> Tuple[List,List]:

	"""
	Parameters
	----------
	filepath: Path of the datafile

	Returns
	-------
	sentences: List of sentences with list of words
	tags : list of tags
	"""

	sentences = []
	tags = []

	sent = ['<S>']
	tag = ['<S>']

	with open(filepath,'r',encoding='utf8') as f:
		for i,line in  enumerate(f):
			if i == 0:
				pass
			else:
				if line=='\n':
					if len(sent)>0:
						sentences.append(sent+['<E>'])
						tags.append(tag+['<E>'])

					sent = ['<S>']
					tag = ['<S>'] 
				else:
					line_s = line.split()
					sent.append(line_s[0])
					tag.append(line_s[-1])

	return sentences,tags

def yield_batches(data: Tuple[List,List],batch_size: int=32,shuffle: bool = True) -> Tuple[List,List]:

	"""
	Parameters
	----------
	data: List of sentences containing list of words and tags

	batch_size: size of batch

	shuffle : shuffling before load into the model


	Returns
	-------
	sentences: List of sentences with list of words
	tags : list of tags
	"""
	data_size  = len(data)
	indices = list(range(data_size))

	if shuffle:
		random.shuffle(indices)

	num_batches = (data_size+batch_size-1)//batch_size

	for i in range(num_batches):
		batch = [data[idx] for idx in indices[i * batch_size: (i + 1) * batch_size]]
		batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
		sentences = [x[0] for x in batch]
		tags = [x[1] for x in batch]
		yield sentences, tags





def main():
	filepath = '/home/ubuntu/Desktop/ExentAI/textofia_web/name_entity_recognition/data/conllpp/conllpp_test.txt'

	s,t = load_dataset(filepath)
	print(len(s),len(t),type(s))
	y= yield_batches(load_dataset(filepath))
	for i in y:
		print(i)
		print()



if __name__ == '__main__':
	main()