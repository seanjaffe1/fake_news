"""
Adopted from code written by Hager Rady and Mo'men AbdelRazek
"""
import argparse

from Agents import *

from utils import process_config

def train(train_file, valid_file, test_file, output_file):

    config = process_config('config.json')

    config.train_file = train_file
    config.valid_file = valid_file
    config.test_file = test_file
    config.out_file = output_file
    
    agent = NLPAgent(config)

    #agent.validate()
    agent.train()

    agent.validate()

    #agent.test(output_file)

def main():

    #train('Data/train.tsv', 'Data/valid.tsv', 'Data/test.tsv', 'predictions.txt')
    train('Data/train.tsv', 'Data/valid.tsv', 'Data/test.tsv', 'predictions.txt')


if __name__ == '__main__':
    main()