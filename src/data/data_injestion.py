import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

logger = logging.getLogger('data_injestion')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler = logging.FileHandler('error.log')
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_params(params_path :str)-> float: 
    try:
        test_size = yaml.safe_load(open(params_path,'r'))['data_injestion']['test_size']
        return test_size
    except ModuleNotFoundError:
        logger.debug('pip install yaml')
        raise
    except FileNotFoundError:
        logger.debug('cannot locate the file')
        raise
    except Exception as e:
        logger.debug(e)
        raise


def read_data(url : str)-> pd.DataFrame:
    df = pd.read_csv(url)
    return df

def process(df):
    df.drop(columns=['tweet_id'],inplace=True)
    final_df = df[df['sentiment'].isin(['happiness','sadness'])]
    final_df.loc[:,'sentiment'] = final_df.loc[:,'sentiment'].replace({'happiness':1, 'sadness':0})
    return final_df

def save_data(data_path : str,train_data : pd.DataFrame,test_data : pd.DataFrame) -> None:
    
    os.makedirs(data_path)    
    train_data.to_csv(os.path.join(data_path,'train.csv'))
    test_data.to_csv(os.path.join(data_path,'test.csv'))





def main():
    test_size = load_params(params_path = 'params.yaml')
    df = read_data(url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    final_df = process(df)
    
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
    data_path = os.path.join('data','raw')
    save_data(data_path=data_path,train_data=train_data,test_data=test_data)

if __name__ == '__main__':
    main()
