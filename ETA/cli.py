import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

import argparse

from analyzer import Analyzer
from labelled_analyzer import LabelledAnalyzer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETA/Analyzer: A text analysis tool.")
    # input arguments
    parser.add_argument(
        "--input",
        required=True,
        help="Input file path. Input file should be a txt or csv file.",
    )
    parser.add_argument(
        "--labelled",
        default=False,
        help="If you data is labelled, set this flag to True.",
    )
    parser.add_argument(
        "--text_column",
        default="text",
        help="Name of the column that contains text. Only used for csv files.",
    )
    parser.add_argument(
        "--delimiter",
        default="\n",
        help="Delimiter that is used to split documents. Only used for txt files.",
    )
    parser.add_argument(
        "--label_column",
        help="To have analysis results for each label, provide the name of the column that contains labels. Only used for csv files. --labelled flag should be set to True.",
    )
    parser.add_argument(
        "--label_separator",
        default="\t",
        help="Delimiter that is used to split labels. Only used for txt files. --labelled flag should be set to True.",
    )
    # analysis arguments
    parser.add_argument(
        "--stopwords",
        default="english",
        help="Stopwords language. Stopwords will be downloaded from nltk.",
    )
    parser.add_argument(
        "--n_nonalpha",
        default=10,
        help="Number of most common non-alphanumeric characters in your dataset to be shown.",
    )
    parser.add_argument(
        "--ngram_nrange",
        default=(1, 3),
        help="Range of n values for n-grams. Tuple of two integers.",
    )
    parser.add_argument(
        "--ngram_firstk", default=10, help="Number of most common n-grams to be shown."
    )
    parser.add_argument(
        "--n_disrtibution_bins",
        default=10,
        help="Number of bins for distribution plots.",
    )
    # output arguments
    parser.add_argument(
        "--output_folder",
        default="stats",
        help="Output folder name. Output folder will be created if it does not exist.",
    )
    parser.add_argument(
        "--file_type", default=True, help="Save stats as json and txt files."
    )
    parser.add_argument(
        "--stats_type", default="json", help="Stats file type. Can be json or txt."
    )
    parser.add_argument("--save_plots", default=True, help="Save plots as png files.")

    args = parser.parse_args()
    if args.input[-4:] not in [".txt", ".csv"]:
        raise ValueError("Input file should be a txt or csv file.")

    if args.labelled and args.labelled.lower() == "true":
        args.labelled = True
    else:
        args.labelled = False

    if args.labelled:
        analyzer = LabelledAnalyzer(
            stopwords=args.stopwords,
            n_nonalpha=args.n_nonalpha,
            ngram_nrange=args.ngram_nrange,
            ngram_firstk=args.ngram_firstk,
            n_disrtibution_bins=args.n_disrtibution_bins,
        )
        if args.input.endswith(".csv"):
            analyzer.read_csv(
                args.input, text_column=args.text_column, label_column=args.label_column
            )
        elif args.input.endswith(".txt"):
            analyzer.read_txt(
                args.input,
                delimiter=args.delimiter,
                label_separator=args.label_separator,
            )
    else:
        analyzer = Analyzer(
            stopwords=args.stopwords,
            n_nonalpha=args.n_nonalpha,
            ngram_nrange=args.ngram_nrange,
            ngram_firstk=args.ngram_firstk,
            n_disrtibution_bins=args.n_disrtibution_bins,
        )
        if args.input.endswith(".csv"):
            analyzer.read_csv(args.input, text_column=args.text_column)
        elif args.input.endswith(".txt"):
            analyzer.read_txt(args.input, delimiter=args.delimiter)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if args.file_type == "json":
        analyzer.to_json(folder_name=args.output_folder)
    elif args.file_type == "txt":
        analyzer.to_txt(folder_name=args.output_folder)
    else:
        raise ValueError("Stats file type should be json or txt.")

    if args.save_plots:
        analyzer.generate_distribution_plots(
            show=False,
            save=True,
            output_path=args.output_folder + "/distribution_plots",
        )
        analyzer.generate_ngram_plots(
            show=False, save=True, output_path=args.output_folder + "/ngram_plots"
        )
        analyzer.generate_word_cloud(output_path=args.output_folder)
