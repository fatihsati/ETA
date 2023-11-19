
"""
    To read data from both txt and csv. Also should be able to read it as str.
"""
import pandas as pd


class FileManager:

    def read_csv(self, path, column_name='text'):
        
        df = pd.read_csv(path)
        if column_name != 'text':
            df.rename(columns={column_name: 'text'})
            
        return df[[column_name]]

    def read_txt(self, path, delimiter='\n'):

        with open(path, mode="r", encoding="utf-8") as f:
            data = f.read()
        data = data.split(delimiter)
        data = [doc.strip() for doc in data if doc.strip()] # eliminate empty lines, strip docs.
        df = pd.DataFrame({'text': data})
        return df

    #TODO to_csv
    #TODO to_json
    #TODO to_txt

