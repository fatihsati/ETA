from text_analyzer import Analyzer
from text_analyzer import LabelledAnalyzer
import pandas as pd

df = pd.read_csv('movie_reviews.csv', encoding='utf-8')

# analyzer = Analyzer()
# analyzer.read_df(df, text_column='review')
# analyzer.read_csv('movie_reviews.csv', text_column='review')
# analyzer.read_txt('movie_reviews.txt', delimiter='\n')
# analyzer.print_stats()
# analyzer.to_json('val_stats')
# analyzer.to_txt('val_stats.txt')

analyzer = LabelledAnalyzer()
analyzer.read_txt('movie_reviews.txt', delimiter='\n', label_separator='\t')
# analyzer.print_stats('pos')
# analyzer.to_txt('stats')