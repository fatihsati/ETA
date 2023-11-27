import pandas as pd
import json

class FileManager:

    def _load_csv(self, path, column_name='text', encoding='utf-8'):
        
        df = pd.read_csv(path, encoding=encoding)
        if column_name != 'text':
            df.rename(columns={column_name: 'text'}, inplace=True)
            
        return df[["text"]]

    def _load_txt(self, path, delimiter='\n'):

        with open(path, mode="r", encoding="utf-8") as f:
            data = f.read()
        data = data.split(delimiter)
        data = [doc.strip() for doc in data if doc.strip()] # eliminate empty lines, strip docs.
        df = pd.DataFrame({'text': data})
        return df

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


