from text_analyzer.analyzer import Analyzer
from text_analyzer.file_manager import FileManager
import os

class LabelledAnalyzer(FileManager):

    def __init__(self):
        self.classes = []
        self.data = None


    def read_csv(self, path: str, text_column:str='text', label_column='label',
                  encoding='utf-8'):
        self.data = self._load_csv(path, text_column, label_column, encoding)

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
        >>> analyzer.read_txt('movie_reviews.txt', delimiter='\n', label_separator='\t')
        """
        self.data = self._load_txt(path, delimiter, label_separator)
        self.analyze()


    def _get_classes(self):
        """Get unique classes from data."""
        classes = self.data['label'].unique()
        return classes
    
    def _analyze_classes(self):
        """Loop through every class and create an analyzer object for each class. Then analyze it. Return a dictionary with class names as keys and analyzer objects as values."""
        data = {}
        for class_ in self.classes:
            analyzer = Analyzer()
            analyzer.read_df(self.data[self.data['label']==class_])
            data[class_] = analyzer
        return data
    
    def analyze(self):
        """Get class names and analyze each class."""
        self.classes = self._get_classes()
        self.data = self._analyze_classes()
    
    def _check_if_data_loaded(self):
        """Check if data is loaded."""
        if not self.data:
            raise ValueError("Use read_csv() or read_txt() methods to read data first.")
        
    def print_stats(self, class_name):
        """print stats for a given class."""
        self._check_if_data_loaded()

        if class_name not in self.classes:
            raise ValueError(f"Class {class_name} not found in the data.")
        self.data[class_name].print_stats()

    def _get_filename_list(self, filename_list):
        if filename_list:
            if len(filename_list) != len(self.classes):
                raise ValueError(f"Length of filename_list should be equal to the number of classes. You have {len(filename_list)} filenames for {len(self.classes)} classes.")
        else:   # if filename_list is not given, use class names as filenames
            filename_list = self.classes
        return filename_list
    
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
            self.data[class_].to_json(path)
    
    def to_txt(self, folder_name:str='stats', filename_list:list=None):
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
            self.data[class_].to_txt(path)