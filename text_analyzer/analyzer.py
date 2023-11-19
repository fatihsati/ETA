from file_manager import FileManager


class Analyzer:

    def __init__(self, data):
        self.data = data

    def calculate_simple_stats(self, row):
        data['n_char'] = len(row['text'])
        data['n_word'] = len(row['text'].split())

        
    def get_simple_stats(self):
        self.data['n_char'] = self.data['text'].apply(lambda x: len(x))
        self.data['n_word'] = self.data['text'].apply(lambda x: len(x.split()))
        char_counts_data = self.data['n_char'].agg(['min', 'mean', 'max']).to_dict()
        word_counts_data = self.data['n_word'].agg(['min', 'mean', 'max']).to_dict()
        #TODO ways to get number for ï like letters.
        #TODO: # of non-letter 
        #TODO: # of links
        #TODO: # of phone numbers

fm = FileManager()
data = fm.read_txt('/Users/fatih/Desktop/val.txt')
print(data.iloc[-1, :])