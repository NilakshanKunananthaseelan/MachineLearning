
from typing import Dict,List
from collections import Counter
from itertools import chain
import json

from utils import load_dataset

class Vocab:
	UNK: str =  '<UNK>'
	PAD: str = '<PAD>'
	START: str = '<S>'
	END: str = '<E>'

	def __init__(self,word2id,id2word):
		self._word2id = word2id
		self._id2word = id2word

	def _get_word2id(self)-> Dict:
		""" Returns the word2id dictionary"""
		"""
		Parameters
		----------
		None

		Returns
		-------
		word2id : dictionary
		Dictionary maps words of dataset to unique integer id.
		"""
		return self._word2id

	def _get_id2word(self)->List:
		""" Return the id2 word dictionary"""

		"""
		Parameters
		----------
		None

		Returns
		-------
		id2word : List
		Dictionary maps integer id unique words
		"""
		return self._id2word

	def id2word(self,idx :int):
		"""Return word for the given index"""

		"""
		Parameters
		----------
		idx : integer
		Value to identify a word

		Returns
		-------
		word: string
		Word from the vocabulary corresponding to idx
		"""

		return self._id2word[idx]

	@staticmethod
	def build_vocab(data: List[List[str]],max_dict_size: int,drop_th: int, is_tags: bool):
		"""
		Parameters
		----------
		data: List of sentences containing list of words

		max_dict_size: maximum size of dictionary

		drop_th : minimum number of occurences to keep the word

		is_tags: build vocab for sentences or tags

		Returns
		-------
		word2id: Dictionary
		id2word: List
		"""

		word_counts_dict = Counter(chain(*data))
		valid_words_list = [word for word,count in word_counts_dict.items() if count>=drop_th]
		valid_words_sorted = sorted(valid_words_list, key=lambda x: word_counts_dict[x], reverse=True)
		valid_words_list = valid_words_list[: max_dict_size]
		valid_words_list += ['<PAD>']

		word2id_dict = {word:idx for idx,word in enumerate(valid_words_list)}


		if not is_tags:
			word2id_dict['<UNK>'] = len(word2id_dict)

		id2word = [word for word,idx in word2id_dict.items()]
		return Vocab(word2id_dict,id2word)


	def save(self, file_path: str)-> None:
		"""Save the vocabulary as json object"""
		with open(file_path, 'w', encoding='utf8') as f:
			json.dump({'word2id': self._word2id, 'id2word': self._id2word}, f, ensure_ascii=False)

	@staticmethod
	def load(file_path: str)->None:
		"""Load the vocab dictionary from json file"""
		with open(file_path, 'r', encoding='utf8') as f:
			json_file= json.load(f)
		return Vocab(word2id=json_file['word2id'], id2word=json_file['id2word'])

	@staticmethod
	def sentence_to_indices(sentence: List[List[str]],vocab)->List[List[int]]:

		"""
		Parameters
		----------
		sentence: List words

		vocab:Vocab instance of word2id

		load_vocab : load vocab from json

		Returns
		-------
		sentences:List of words mapped to integers
		"""
		if isinstance(sentence[0],list):
			vector = [[vocab._word2id[word] if word in vocab._word2id else vocab._word2id['<UNK>'] for word in sent] for sent in sentence]
		else:
			vector = [vocab._word2id[word] if word  in vocab._word2id else vocab._word2id['<UNK>']  for word in sentence]

		return vector

	@staticmethod
	def indices_to_words(sentence: List[List[str]],vocab)->List[List[int]]:

		"""
		Parameters
		----------
		sentence: List sentences mapped to integers

		vocab:Vocab instance

		load_vocab : load vocab from json

		Returns
		-------
		sentences:List of words 
		"""
		if isinstance(sentence[0],list):
			vector = [[vocab._id2word[word] if word in vocab._id2word else vocab._id2word[vocab._word2id['<UNK>']] for word in sent] for sent in sentence]
		else:
			vector = [vocab._id2word[word] if word in vocab._id2word else vocab._id2word[vocab._word2id['<UNK>']]  for word in sentence]

		return vector



def main()->None:

	path = '/home/ubuntu/Desktop/ExentAI/textofia_web/name_entity_recognition/data/conllpp/conllpp_test.txt'


	sentences, tags = load_dataset(path)
	sent_vocab = Vocab.build_vocab(sentences, int(50), int(0), is_tags=False)
	tag_vocab = Vocab.build_vocab(tags, int(50), int(0), is_tags=True)
	sent_vocab.save('sent.json')
	tag_vocab.save('tags.json')

	print(sent_vocab._word2id)

	
	Vocab.sentence_to_indices(sentences,sent_vocab)


if __name__ == '__main__':
	main()








