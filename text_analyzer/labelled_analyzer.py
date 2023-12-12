from text_analyzer.analyzer import Analyzer
from text_analyzer.file_manager import FileManager
from text_analyzer.plotter import Plotter
import pandas as pd
import os


class LabelledAnalyzer(FileManager):

    def __init__(self):
        self.classes = []
        self.data = None
        self.analyze_objects = None
        self.num_classes = None
        self.num_docs = None

    def __str__(self):
        if isinstance(self.data, pd.DataFrame):
            return f"LabelledAnalyzer(num_classes={self.num_classes}, num_docs={self.num_docs}). \
                Use print_stats() method to see stats or use to_json() or to_txt() methods to save stats."
        else:
            return "LabelledAnalyzer(). Use read_csv() or read_txt() methods to read data first."
    
    def __repr__(self):
        if isinstance(self.data, pd.DataFrame):
            return f"LabelledAnalyzer(num_classes={self.num_classes}, num_docs={self.num_docs}). \
                Use print_stats() method to see stats or use to_json() or to_txt() methods to save stats."
        else:
            return "LabelledAnalyzer(). Use read_csv() or read_txt() methods to read data first."

    def read_csv(self, path: str, text_column:str='text', label_column='label',
                  encoding='utf-8'):
        """Load data from a csv file.

    Parameters
    ----------
        path : str
            Path to csv file.
        text_column : str
            Name of the column that contains text. The default is 'text'.
        label_column : str
            Name of the column that contains labels. The default is 'label'.
        encoding : str
            Encoding of the csv file. The default is 'utf-8'.

    Example
    -------
        >>> analyzer = LabelledAnalyzer()
        >>> analyzer.read_csv('movie_reviews.csv', text_column='review', label_column='sentiment')
        """
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
            Delimiter to split labels and text. Labels have to be at the beggining \
                of the sentence. The default is '\\t'.

    Example
    -------
        >>> analyzer = LabelledAnalyzer()
        >>> analyzer.read_txt('movie_reviews.txt', delimiter='\\n', label_separator='\\t')
        """
        self.data = self._load_txt(path, delimiter, label_separator)
        self._analyze()

    def read_df(self, df: pd.DataFrame, text_column:str='text', label_column='label'):
        """
    Parameters
    ----------
        df : pandas DataFrame
            DataFrame that contains text and label columns.
        text_column : str
            Name of the column that contains text. The default is 'text'.
        label_column : str
            Name of the column that contains labels. The default is 'label'.
    
    Example
    -------
        >>> analyzer = LabelledAnalyzer()
        >>> analyzer.read_df(df, text_column='review', label_column='sentiment')
        """
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

    def generate_plots(self, classes:list=None, add_class_distribution=True, show=True, save=False, \
                       output_name='stats', return_plot=False, return_list=False):
        """Generates a plot for word and char distribution for classes. If classes is not given, all classes will be plotted.

    Parameters
    ----------
        classes : list, default None
            List of classes to plot. If not given, all classes will be plotted.
        add_class_distribution : bool, default True
            If True, class distribution will be added to the plot.
        show : bool, default True
            If True, plot will be shown automatically.
        save : bool, default False
            If True, plot will be saved, name will be 'output_name'.
        output_name : str, default 'stats'
            Name of the output file. If save is True, plot will be saved as output_name.
        return_plot : bool, default False
            If True, plot will be returned as a matplotlib figure at the end of the function.
        return_list : bool, default False
            If True, intervals and frequencies will be returned as a list of dictionaries at the end of the function.
        # return_plot and return_list cannot be used at the same time.
    
    Example
    -------
        >>> analyzer = LabelledAnalyzer()
        >>> analyzer.read_csv('movie_reviews.csv', text_column='review', label_column='sentiment')
        >>> analyzer.generate_plots(add_class_distribution=False, show=True, save=True, output_name='stats')
        """

        self._check_if_data_loaded()
        classes = self._check_given_classes(classes)

        plotter = Plotter()
        word_distribution_list = self._get_word_distribution(classes)
        char_distribution_list = self._get_char_distribution(classes)
        
        series_list = [s for pair in zip(word_distribution_list, char_distribution_list) for s in pair]

        if add_class_distribution:
            class_distribution = self._get_class_distribution()
            series_list.append(class_distribution)

        plot = plotter.generate_plots_from_series(*series_list)
        if save: plot.savefig(output_name)
        if show: plot.show()

        if return_plot and return_list:
            raise ValueError("return_plot and return_list cannot be True at the same time.")
        
        if return_plot: return plot
        elif return_list: return [s.to_dict() for s in series_list]

        return None

    
    def to_json(self, folder_name:str='stats', filename_list=None):
        """
        Save stats for every class in a new json file.

        Parameters
        ----------
        folder_name : str, default 'stats'
            Folder name to save JSON files. If folder does not exist, it will be created.
        filename_list : list, default None
            List of filenames. If not given, class names will be used as filenames.

        Example
        -------
        >>> analyzer = LabelledAnalyzer()
        >>> analyzer.read_csv('movie_reviews.csv', text_column='review', label_column='sentiment')
        >>> analyzer.to_json('stats')

        """
        self._check_if_data_loaded()

        filename_list = self._get_filename_list(filename_list)
        
        if not os.path.exists(folder_name):
            os.mkdir(folder_name) # create folder if it doesn't exist

        # loop through classes and save stats as json files
        for class_, filename in zip(self.classes, filename_list):
            path = self._generate_path_from_filename(folder_name, filename, extension='.json')
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

        Example
        -------
        >>> analyzer = LabelledAnalyzer()
        >>> analyzer.read_csv('movie_reviews.csv', text_column='review', label_column='sentiment')
        >>> analyzer.to_txt('stats', filename_list=['neg', 'pos'])
        """
        self._check_if_data_loaded()

        filename_list = self._get_filename_list(filename_list)
        
        if not os.path.exists(folder_name):
            os.mkdir(folder_name) # create folder if it doesn't exist

        # loop through classes and save stats as json files
        for class_, filename in zip(self.classes, filename_list):
            path = self._generate_path_from_filename(folder_name, filename, extension='.txt')
            self.analyze_objects[class_].to_txt(path)

    def _analyze(self):
        """Get class names and analyze each class."""
        self.classes = self._get_classes()
        self.analyze_objects = self._analyze_classes()
        self.num_classes = self._get_number_of_classes()
        self.num_docs = self._get_number_of_documents()

    def _analyze_classes(self):
        """Loop through every class and create an analyzer object for each class. Then analyze it. \
            Return a dictionary with class names as keys and analyzer objects as values."""
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
        """Get filename list. If not given, use class names as filenames."""
        if filename_list:
            if len(filename_list) != len(self.classes):
                raise ValueError(f"Length of filename_list should be equal to the number of \
                                 classes. You have {len(filename_list)} filenames for {len(self.classes)} classes.")
        else:   # if filename_list is not given, use class names as filenames
            filename_list = self.classes
        return filename_list
    
    def _generate_path_from_filename(self, folder_name, filename, extension='.json'):
        """Generate a path from filename. If filename exists, add _1 to the end of the filename.\
            Return a path.
        """
        filename = str(filename)
        filename = filename + extension if not filename.endswith(extension) else filename
        path = os.path.join(folder_name, str(filename))
        if os.path.exists(path):
            path = path[:-len(extension)] + '_1' + extension
        return path
    
    def _get_class_distribution(self):
        return self.data['label'].value_counts()

    def _get_char_distribution(self, classes, num_bins=10):
        """Get char distribution for every class. Return a list of Series indicating char distribution for every class. 
        Class names are added to index names."""
        char_bins = self._calculate_bins_for_char()
        char_distribution_list = []
        for class_ in classes:
            distribution_series = self.analyze_objects[class_]._get_char_distribution(num_bins=num_bins, bins=char_bins)
            distribution_series.index.name = str(class_) + ' ' + distribution_series.index.name
            char_distribution_list.append(distribution_series)

        return char_distribution_list

    def _get_word_distribution(self, classes, num_bins=10):
        """Get word distribution for every class. Return a list of Series indicating char distribution for every class. 
        Class names are added to index names."""
        word_bins = self._calculate_bins_for_word()
        word_distribution_list = []
        for class_ in classes:
            distribution_series = self.analyze_objects[class_]._get_word_distribution(num_bins=num_bins, bins=word_bins)
            distribution_series.index.name = str(class_) + ' ' + distribution_series.index.name
            word_distribution_list.append(distribution_series)

        return word_distribution_list
    
    def _calculate_bins_for_char(self):
        self.data['n_char'] = self.data['text'].apply(lambda x: len(x))
        word_bins = self.analyze_objects[self.classes[0]]._calculate_bins(self.data, column_name='n_char')

        return word_bins

    def _calculate_bins_for_word(self):
        self.data['n_word'] = self.data['text'].apply(lambda x: len(x.split()))
        word_bins = self.analyze_objects[self.classes[0]]._calculate_bins(self.data, column_name='n_word')
        
        return word_bins

    def _check_given_classes(self, classes):
        if classes is None: # if not given, use all classes
            return self.classes

        if not isinstance(classes, list): # if given, check if it's a list
            classes = [classes]
        
        for item in classes: # check list elements are in classes
            if item not in self.classes:
                raise ValueError(f"Class {item} not found in the data.\n\
                                 Available classes: {self.classes}, dtype: {self.classes.dtype}")
        return classes
