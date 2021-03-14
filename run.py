import argparse
import pickle

import yaml

from tagger import pre_process, model
import os, sys


def main(args):
    mode = args.mode
    with open(args.config, "r") as yaml_in:
        config = yaml.load(yaml_in, Loader=yaml.FullLoader)
    if mode == "train":
        print(f"Training a model on {config['train_file']}.")
        X, Y = pre_process.load_dataset(config["train_file"])
        X, Y = pre_process.prepare_data_for_training(X, Y)
        tagger = model.POSTagger()
        tagger.fit_and_report(X, Y, config["crossval"], config["n_folds"])
        tagger.save_model(config["model_file"])
    elif mode == "tag":
        print(f"Tagging text using pretrained model: {config['model_file']}.")
        with open(config["model_file"], "rb") as model_in:
            tagger = pickle.load(model_in)
        if os.path.exists(args.text):
            print('Reading file...')
            writeFile = True
            with open(args.text, 'r') as txt_file:
                text = txt_file.read()
                tagged_sent = tagger.tag_sentence(text)
        else:
            writeFile = False
            tagged_sent = tagger.tag_sentence(args.text)

        output = [f"{token[0]}\t{token[1]}".expandtabs(15) for token in tagged_sent]
        if writeFile:
            with open(args.text + '.tag', 'w') as output_file:
                output_file.writelines(output)
        else:
            print('\n'.join(output))


    else:
        print(f"{args.mode} is an incompatible mode. Must be either 'train' or 'tag'.")


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description="A basic SVM-based POS-tagger. \
                                     Accepts either .conllu or tab-delineated \
                                     .txt files for training.")

    PARSER.add_argument('--mode', metavar='M', type=str,
                        help="Specifies the tagger mode: {train, tag, eval}.")
    PARSER.add_argument('--text', metavar='T', type=str,
                        help="Tags a sentence string. \
                        Can only be called if '--mode tag' is specified.")
    PARSER.add_argument('--config', metavar='C', type=str,
                        help="A config .yaml file that specifies the train data, \
                        model output file, and number of folds for cross-validation.")

    ARGS = PARSER.parse_args()
    main(ARGS)
