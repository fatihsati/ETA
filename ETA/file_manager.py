import pandas as pd
import json

class FileManager:

    def _load_csv(self, path, text_column='text', label_column=None, encoding='utf-8'):
        
        df = pd.read_csv(path, encoding=encoding)
        if text_column in df.columns:
            if text_column != 'text':
                df.rename(columns={text_column: 'text'}, inplace=True)
        else:
            raise ValueError(f"Text column {text_column} not found in the dataframe.")

        if label_column:
            if label_column in df.columns:
                if label_column != 'label':
                    df.rename(columns={label_column: 'label'}, inplace=True)
            else:
                raise ValueError(f"Label column {label_column} not found in the dataframe.")
     
        return df

    def _load_txt(self, path, delimiter='\n', label_separator=None):
        """Load txt, use demiliter to split docs. If label_separator is given, split docs into label and text. Labels should be at the beginning of the line. Remove empty lines."""

        with open(path, 'r', encoding='utf-8') as f:
            data = f.read()
        
        data = data.split(delimiter)
        if label_separator:
            data = [doc.split(label_separator) for doc in data]
            data = pd.DataFrame(data, columns=['label', 'text'])
        else:
            data = pd.DataFrame(data, columns=['text'])
        
        data = data[data['text'] != '']
        data = data.reset_index(drop=True)
        data = data.dropna()
        return data
        
    def _to_json(self, data, output_name: str):
        
        if not output_name.endswith('.json'):
            output_name+='.json'
        
        with open(output_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _to_txt(self, data, output_name):
        
        if not output_name.endswith('.txt'):
            output_name+='.txt'

        with open(output_name, 'w') as file:
            self._write_dict(data, file, 0)

    def _write_dict(self, data, file, indent):
        for key, value in data.items():
            if isinstance(value, dict):
                file.write(' ' * indent + str(key) + '\n')
                self._write_dict(value, file, indent + 4)
            else:
                file.write(' ' * indent + f"{key}: {value}\n")

    def _generate_json_output(self, analyzer_dict, pretty):
        return json.dumps(analyzer_dict, indent=4 if pretty else None)
        