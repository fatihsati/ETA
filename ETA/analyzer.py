from ETA.file_manager import FileManager
from ETA.plotter import Plotter
from collections import Counter
from nltk.util import ngrams
import pandas as pd
import numpy as np
import re


class Analyzer(FileManager):

    def __init__(self):
        self.data = None
        self.document_count = None
        self.char_number = None
        self.word_number = None
        self.non_alpha_chars = None
        self.ngram_dict = None
        
    def __str__(self):
        if isinstance(self.data, pd.DataFrame):
            return f"Analyzer(num_docs={self.document_count}). Use print_stats() method to see stats or use to_json() or to_txt() methods to save stats."
        else:
            return "Analyzer(). Use one of the read_csv(), read_txt() or read_df() methods to read data first."
    
    def __repr__(self):
        if isinstance(self.data, pd.DataFrame):
            return f"Analyzer(num_docs={self.document_count}). Use print_stats() method to see stats or use to_json() or to_txt() methods to save stats."
        else:
            return "Analyzer(). Use one of the read_csv(), read_txt() or read_df() methods to read data first."
    
    def read_csv(self, path: str, text_column:str='text', encoding='utf-8'):
        """Load data from csv file
        
    Parameters
    ----------
        path : str
            Path of the csv file.
        text_column : str
            Name of the column that contains text, by default 'text'
        encoding : str
            Encoding of the csv file, by default 'utf-8'
        
    Example
    --------
        >>> analyzer = Analyzer()
        >>> analyzer.read_csv('movie_reviews.csv', text_column='review')
    """

        self.data = self._load_csv(path=path, text_column=text_column, encoding=encoding)
        self.data = self.data['text'].astype(str)
        self.analyze()

    def read_txt(self, path, delimiter='\n'):
        """Load data from txt file
    
    Parameters
    ----------
        path : str
            Path of the txt file.
        delimiter : str
            Delimiter that is used to split documents, by default '\\n'
        
    Example
    --------
        >>> analyzer = Analyzer()
        >>> analyzer.read_txt('movie_reviews.txt', delimiter='\\n')
    """

        self.data = self._load_txt(path, delimiter)
        self.analyze()

    def read_df(self, df: pd.DataFrame, text_column:str='text'):
        """Load data from pandas DataFrame

    Parameters
    ----------
        df : pandas DataFrame
            Dataframe that contains text.
        text_column : str
            Name of the column that contains text, by default 'text'
        
    Example
    --------
        >>> analyzer = Analyzer()
        >>> analyzer.read_df(df, text_column='review')    
    """
        
        if text_column not in df.columns:
            raise ValueError(f"Text column {text_column} not found in the dataframe.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Data should be a pandas DataFrame.")
        if text_column != 'text':
            df.rename(columns={text_column: 'text'}, inplace=True)
        
        df['text'] = df['text'].astype(str)
        self.data = df
        self.analyze()

    def analyze(self):
        """Analyze data and update class attributes."""
        self._check_if_data_loaded()
        self._preprocess_data()
        self.document_count = self._get_document_count()
        self.char_number = self._calculate_char_number()
        self.word_number = self._calculate_word_number()
        self.non_alpha_chars = self._count_non_alpha_chars()
        self.ngram_dict = self._calculate_most_used_ngrams()

    def _preprocess_data(self):
        """Preprocess data. Remove punctuations, numbers, whitespaces. Lowercase.\
            Add a new column named 'processed_text' to the dataframe.\
                Update self.data with the new dataframe."""
        self.data['processed_text'] = self.data['text'].apply(self._preprocessing)
    
    def _preprocessing(self, text: str):
        """lower, remove punctuations, remove numbers, remove whitespaces. Use regex."""
        text = text.lower()

        text = re.sub(r'[^\w\s]','', text)
        text = re.sub(r'\d','', text)
        text = re.sub(r'\s+',' ', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n', ' ', text)
        return text.strip()
    
    def _calculate_char_number(self):
        """calculate number of chars for each doc. Return min, max, mean values as dict. \
            Include longest and shortest 5 docs's char counts."""
        self.data['n_char'] = self.data['text'].apply(lambda x: len(x))
        agg_values = self.data['n_char'].agg(['min', 'mean', 'max', 'sum']).to_dict()
        n_char_sorted = self.data['n_char'].sort_values()
        N = min(5, len(n_char_sorted))
        top_N = n_char_sorted.tail(N).tolist()
        bottom_N = n_char_sorted.head(N).tolist()
        agg_values['longest_doc_lengths'] = top_N
        agg_values['shortest_doc_lengths'] = bottom_N
        return agg_values
    
    def _calculate_word_number(self):
        """calculate number of words for each doc. Return min, max, mean values as dict. \
            Include longest and shortest 5 docs's word counts."""
        self.data['n_word'] = self.data['text'].apply(lambda x: len(x.split()))
        agg_values = self.data['n_word'].agg(['min', 'mean', 'max', 'sum']).to_dict()
        n_word_sorted = self.data['n_word'].sort_values()
        N = min(5, len(n_word_sorted))
        top_N = n_word_sorted.tail(N).tolist()
        bottom_N = n_word_sorted.head(N).tolist()
        agg_values['longest_doc_lengths'] = top_N
        agg_values['shortest_doc_lengths'] = bottom_N
        return agg_values
    
    def _count_non_alpha_chars(self, n=10):
        non_alpha_chars = Counter()
        total_non_alpha_count = 0
        
        for text in self.data['text']:
            non_alpha = [char for char in text if not char.isalpha() and not char.isspace()]
            non_alpha_chars.update(non_alpha)
            total_non_alpha_count += len(non_alpha)
        
        return {"10_most_common_with_freq": dict(non_alpha_chars.most_common(n)),
                "total_count": total_non_alpha_count}
    
    def _calculate_most_used_ngrams(self, n_range=(1,3), first_k=10):
        """calculate most used ngrams for each doc. Return dict of ngrams. Keys are n values. \
            Values are dicts of ngrams and their frequencies. n_min and n_max are used to \
                determine n values. first_k is used to determine how many ngrams will be returned."""
        text_splitted = [word for sentence in self.data['processed_text'].tolist() for word in sentence.split()]
        ngram_dict = dict()
        for n in range(n_range[0], n_range[1]+1):
            ngram = Counter(ngrams(text_splitted, n)).most_common(first_k)
            ngram_freq = {" ".join(phrase): freq for phrase, freq in ngram}
            ngram_dict[n] = ngram_freq

        return ngram_dict

    def _get_document_count(self):
        return len(self.data)
    
    def generate_word_cloud(self, use_processed_data=True, save=False, output_name='word_cloud.png'):
        """import wordcloud library and generate a single word cloud with all the documents"""

        plotter = Plotter()
        self._check_if_data_loaded()

        data = self.data['processed_text'].tolist() if use_processed_data else self.data['text'].tolist()
        world_cloud = plotter.get_word_cloud(data, width=800, height=800, background_color='white')
        if save:
            world_cloud.to_file(output_name)
        world_cloud.show()
        # world_cloud = WordCloud(width=800, height=800, background_color='white',
        #                          min_font_size=10).generate(" ".join(data))
        # plt.axis('off')
        # plt.imshow(world_cloud, interpolation='bilinear')
        # if save:
        #     world_cloud.to_file(output_name)
        # plt.show()
    
    def print_stats(self, pretty=True):

        self._check_if_data_loaded()
        analyzer_dict = self.__dict__.copy()
        analyzer_dict.pop('data')
        
        res = self._generate_json_output(analyzer_dict, pretty=pretty)
        print(res)
        
    def to_json(self, output_name: str):
        self._check_if_data_loaded()

        analyzer_dict = self.__dict__.copy()
        analyzer_dict.pop('data')
        self._to_json(analyzer_dict, output_name)

    def to_txt(self, output_name):
        self._check_if_data_loaded()
        analyzer_dict = self.__dict__.copy()
        analyzer_dict.pop('data')
        
        self._to_txt(analyzer_dict, output_name)
    
    def generate_plots(self, show=True, num_bins=10, save=False, output_name='plots.png', 
                       return_plot=False, bins_word=None, bins_char=None):
        self._check_if_data_loaded()
        plotter = Plotter()
        word_distribution = self._get_word_distribution(num_bins=num_bins, bins=bins_word)
        char_distribution = self._get_char_distribution(num_bins=num_bins, bins=bins_char)
        plot = plotter.generate_plots_from_series(word_distribution, char_distribution)
        if save:
            plot.savefig(output_name)
        if show:
            plot.show()
        if return_plot:
            return plot

    def _check_if_data_loaded(self):
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data not loaded. Use read_csv(), read_txt() or read_df() methods to load data first.")
        
    def _get_word_distribution(self, num_bins=10, bins=None):
        
        bins = self._calculate_bins(df=self.data, num_bins=num_bins, column_name='n_word') if bins is None else bins
        
        word_interval = pd.cut(self.data['n_word'], bins=bins)
        return word_interval.value_counts()

    def _get_char_distribution(self, num_bins=10, bins=None):
        bins = self._calculate_bins(df=self.data, num_bins=num_bins, column_name='n_char') if bins is None else bins
    
        char_interval = pd.cut(self.data['n_char'], bins=bins)
        return char_interval.value_counts()
    
    def _calculate_lower_upper_bounds(self, data: pd.Series):
        sorted_data = np.sort(data.values)
        q1 = np.percentile(sorted_data, 25)
        q3 = np.percentile(sorted_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return lower_bound, upper_bound

    def _calculate_bins(self, df: pd.DataFrame=None, num_bins=10, column_name=None):

        lower_bound, upper_bound = self._calculate_lower_upper_bounds(df[column_name])
        outlier_column_name = f'is_outlier_{column_name}'
        df[outlier_column_name] = df[column_name].apply(lambda x: 1 if x < lower_bound or x > upper_bound else 0)
        data = df[df[outlier_column_name] == 0]

        min_count = data[column_name].min()
        max_count = data[column_name].max()
        bin_size = (max_count - min_count) / num_bins
        bin_size = max(bin_size, 1)
        bins = [i for i in range(min_count, max_count + int(bin_size), int(bin_size))]
        if len(bins) > num_bins + 1:
            bins = bins[:-1]
        
        bins[0] = 0
        bins.append(df[column_name].max())
        return bins
    