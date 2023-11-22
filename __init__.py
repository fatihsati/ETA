from text_analyzer.analyzer import Analyzer

analyzer = Analyzer()
analyzer.read_txt('text_analyzer/val.txt')

print(analyzer.data.head())

analyzer.print_stats()
analyzer.to_json('val_stats')