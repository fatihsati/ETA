from text_analyzer.analyzer import Analyzer
from text_analyzer.file_manager import FileManager
from text_analyzer.plotter import Plotter
import os
import pandas as pd
pd.options.mode.copy_on_write = True

class LabelledAnalyzer(FileManager):

    def __init__(self):
        self.classes = []
        self.data = None
        self.analyze_objects = None
        self.num_classes = None
        self.num_docs = None

    def __str__(self):
        if isinstance(self.data, pd.DataFrame):
            return f"LabelledAnalyzer(num_classes={self.num_classes}, num_docs={self.num_docs}). Use print_stats() method to see stats or use to_json() or to_txt() methods to save stats."
        else:
            return "LabelledAnalyzer(). Use read_csv() or read_txt() methods to read data first."
    
    def __repr__(self):
        if isinstance(self.data, pd.DataFrame):
            return f"LabelledAnalyzer(num_classes={self.num_classes}, num_docs={self.num_docs}). Use print_stats() method to see stats or use to_json() or to_txt() methods to save stats."
        else:
            return "LabelledAnalyzer(). Use read_csv() or read_txt() methods to read data first."

    def read_csv(self, path: str, text_column:str='text', label_column='label',
                  encoding='utf-8'):
        self.data = self._load_csv(path, text_column, label_column, encoding)
        self.data['text'] = self.data['text'].astype(str)
        self._analyze()

    def read_txt(self, path: str, delimiter='\n', label_separator='\t'):
        """
    Parameters
    ----------
        path : str
            Path to txt file.
        delimiter : str
            Delimiter to split docs. The default is '\\n'.
        label_separator : str
            Delimiter to split labels and text. Labels have to be at the beggining of the sentence. The default is '\\t'.

    Example
    -------
        >>> analyzer = LabelledAnalyzer()
        >>> analyzer.read_txt('movie_reviews.txt', delimiter='\\n', label_separator='\\t')
        """
        self.data = self._load_txt(path, delimiter, label_separator)
        self._analyze()

    def read_df(self, df: pd.DataFrame, text_column:str='text', label_column='label'):
        if text_column not in df.columns:
            raise ValueError(f"Text column {text_column} not found in the dataframe.")
        if label_column not in df.columns:
            raise ValueError(f"Label column {label_column} not found in the dataframe.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Data should be a pandas DataFrame.")
        if text_column != 'text':
            df.rename(columns={text_column: 'text'}, inplace=True)
        if label_column != 'label':
            df.rename(columns={label_column: 'label'}, inplace=True)
        
        # convert text type to string
        df['text'] = df['text'].astype(str)
        self.data = df
        self._analyze()

    def print_stats(self, class_name):
        """print stats for a given class."""
        self._check_if_data_loaded()

        if class_name not in self.classes:
            raise ValueError(f"Class {class_name} not found in the data.")
        self.analyze_objects[class_name].print_stats()

    def generate_plots(self, classes:list=None, add_class_distribution=True, show=True, save=False, output_name='stats', return_plot=False):
        """Generate single plot for every class. If classes is not given, all classes will be plotted."""
        self._check_if_data_loaded()
        classes = self._check_given_classes(classes)

        plotter = Plotter()
        word_distribution_dict = self._get_word_distribution(classes)
        char_distribution_dict = self._get_char_distribution(classes)
        
        series_list = []
        for class_ in classes:
            word_distribution = word_distribution_dict[class_]
            word_distribution.index.name = str(class_) + ' ' + word_distribution.index.name
            series_list.append(word_distribution)
            char_distribution = char_distribution_dict[class_]
            char_distribution.index.name = str(class_) + ' ' + char_distribution.index.name
            series_list.append(char_distribution)

        if add_class_distribution:
            class_distribution = self._get_class_distribution()
            series_list.append(class_distribution)

        plot = plotter.generate_plots_from_series(*series_list)
        if save:
            plot.savefig(output_name)
        if show:
            plot.show()
        if return_plot:
            return plot
    
    def to_json(self, folder_name:str='stats', filename_list=None):
        """
        Save stats for every class in a new json file.

        Parameters
        ----------
        folder_name : str, default 'stats'
            Folder name to save JSON files. If folder does not exist, it will be created.
        filename_list : list, default None
            List of filenames. If not given, class names will be used as filenames.
        """
        self._check_if_data_loaded()

        filename_list = self._get_filename_list(filename_list)
        
        if not os.path.exists(folder_name):
            os.mkdir(folder_name) # create folder if it doesn't exist

        # loop through classes and save stats as json files
        for class_, filename in zip(self.classes, filename_list):
            filename = str(filename)
            filename = filename + '.json' if not filename.endswith('.json') else filename
            path = os.path.join(folder_name, str(filename))
            if os.path.exists(path):
                path = path[:-5] + '_1.json'
            self.analyze_objects[class_].to_json(path)
    
    def to_txt(self, folder_name:str='stats', filename_list:list|None=None):
        """
        Save stats for every class in a new txt file.

        Parameters
        ----------
        folder_name : str, default 'stats'
            Folder name to save txt files. If folder does not exist, it will be created.
        filename_list : list, default None
            List of filenames. If not given, class names will be used as filenames.
        """
        self._check_if_data_loaded()

        filename_list = self._get_filename_list(filename_list)
        
        if not os.path.exists(folder_name):
            os.mkdir(folder_name) # create folder if it doesn't exist

        # loop through classes and save stats as json files
        for class_, filename in zip(self.classes, filename_list):
            filename = str(filename)
            filename = filename + '.txt' if not filename.endswith('.txt') else filename
            path = os.path.join(folder_name, str(filename))
            if os.path.exists(path):
                path = path[:-4] + '_1.txt'
            self.analyze_objects[class_].to_txt(path)

    def _analyze(self):
        """Get class names and analyze each class."""
        self.classes = self._get_classes()
        self.analyze_objects = self._analyze_classes()
        self.num_classes = self._get_number_of_classes()
        self.num_docs = self._get_number_of_documents()

    def _analyze_classes(self):
        """Loop through every class and create an analyzer object for each class. Then analyze it. Return a dictionary with class names as keys and analyzer objects as values."""
        sub_data_dict = {}
        df = self.data[:]
        for class_ in self.classes:
            analyzer = Analyzer()
            analyzer.read_df(df[df['label']==class_])
            sub_data_dict[class_] = analyzer
        return sub_data_dict
    
    def _get_number_of_documents(self):
        return len(self.data)
    
    def _get_number_of_classes(self):
        return len(self.classes)
    
    def _get_classes(self):
        """Get unique classes from data."""
        return self.data['label'].unique()
    
    def _check_if_data_loaded(self):
        """Check if data is loaded."""
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Use read_csv() or read_txt() methods to read data first.")
        
    def _get_filename_list(self, filename_list):
        if filename_list:
            if len(filename_list) != len(self.classes):
                raise ValueError(f"Length of filename_list should be equal to the number of classes. You have {len(filename_list)} filenames for {len(self.classes)} classes.")
        else:   # if filename_list is not given, use class names as filenames
            filename_list = self.classes
        return filename_list
    
    def _get_class_distribution(self):
        return self.data['label'].value_counts()

    def _get_char_distribution(self, classes):
        """Get char distribution for every class. Return a dictionary with class names as keys and char distribution series as values."""
        char_distribution_dict = {}
        for class_ in classes:
            char_distribution_dict[class_] = self.analyze_objects[class_]._get_char_distribution()
        return char_distribution_dict

    def _get_word_distribution(self, classes):
        """Get word distribution for every class. Return a dictionary with class names as keys and word distribution series as values."""
        word_distribution_dict = {}
        for class_ in classes:
            word_distribution_dict[class_] = self.analyze_objects[class_]._get_word_distribution()
        return word_distribution_dict
    
    def _check_given_classes(self, classes):
        
        if classes is None: # if not given, use all classes
            return self.classes

        # if given, check if it's a list
        if not isinstance(classes, list):
            classes = [classes]
        
        # check list elements are in classes
        for item in classes:
            if item not in self.classes:
                raise ValueError(f"Class {item} not found in the data.\nAvailable classes: {self.classes}, dtype: {self.classes.dtype}")
        return classes
