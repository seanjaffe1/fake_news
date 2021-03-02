import os

import string
from tqdm import tqdm
import numpy as np
import json
from easydict import EasyDict
from pprint import pprint

# utilities for reading files and creating pre-trainied embedding matrices
# Used only once before all training experiments

class Vocab():
    def __init__(self):
        self.words = set()
        self.words.add("<PAD")
        self.word2id = {"<PAD>": 0}
        #self.vectors = 
    
    def get_words():
        return self.words
    
    def get_word2id(self, word):
        return self.word2id[word]

    def add(self, word):
        if word not in self.word2id.keys():
            # 0 does not have a word attached
            self.word2id[word] = len(self.words)
            self.words.add(word)

    def __len__(self):
        return len(self.words) 

def add_file_to_vocab(v, input_file):
    with open(input_file, 'r',  encoding='utf-8') as f:
        for line in f:
            l = line.rstrip('\n').split('\t')
            statement = l[1].translate(str.maketrans('', '', string.punctuation)).replace("-"," ").lower().split(" ")
            
            for i, word in enumerate(statement):
                if has_number(word):
                    statement[i] = "NUMBER"

            for word in statement:
                v.add(word)
    return v

def has_number(input):
    return any(char.isdigit() for char in input)


def generate_embedding_matrix(embedding_file, input_file, emb_size = 50):
    vocab = Vocab()
    vecs = []
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            l = line.split()
            vocab.add(l[0])
            vecs.append(np.array(l[1:]).astype(float))
    
    glove_words = len(vecs)
    vocab = add_file_to_vocab(vocab, input_file)
    new_words = len(vocab) - glove_words
    for i in range(new_words):
        vecs.append(np.random.normal(scale =.5, size = (emb_size)))
    return vocab, vecs


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        """
        TODO 3 logging here
        logging.getLogger("Dirs Creator").info("Creating directories error: {0}".format(err))
        exit(-1)

        """
        print("Error on Dir creator")

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
        
        
        except ValueError as e:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config, config_dict


def process_config(json_file):
    """
    Get the json file then editing the path of the experiments folder, creating the dir and return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    config, _ = get_config_from_json(json_file)
    print(" THE Configuration of your experiment ..")
    pprint(config)
    print(" *************************************** ")
    try:
        config.summary_dir = os.path.join("experiments", config.exp_name, "summaries/")
        config.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoints/")
        config.out_dir = os.path.join("experiments", config.exp_name, "out/")
        create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir])
    except AttributeError as e:
        print("ERROR!!..Please provide the exp_name in json file..")
        exit(-1)
    return config