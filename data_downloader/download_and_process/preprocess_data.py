from tqdm.auto import tqdm
import glob
import pandas as pd
from typer import run
import logging

def create_processed_csv():
    logging.basicConfig(
      filename="preprocess_data.log", 
      filemode="w",
      format="%(name)s - %(levelname)s - %(message)s"
      )
    files = [code for code in glob.iglob('output'+'/**/*.cpp', recursive=True)]
    java_files = [code for code in glob.iglob('output'+'/**/*.java', recursive=True)]
    files.extend(java_files)

    df = pd.DataFrame()
    for i, code in tqdm(enumerate(files), total=len(files), desc="Preprocessing data"):
        try:
            with open(code, 'r') as w:
                try:
                    df.loc[i, 'content'] = w.read()
                except UnicodeDecodeError:
                    df.loc[i, 'content'] = w.read().encode('utf8').decode('utf8')

                if code.split('.')[-1]=='cpp':
                    df.loc[i, 'language'] = 'cpp'
                elif code.split('.')[-1]=='java':
                    df.loc[i, 'language']='java'

        except KeyboardInterrupt:
            break
        
        except Exception as e:
            logging.error(e)

    df.to_csv("raw_data.csv", index=False)

if __name__=="__main__":
    run(create_processed_csv)
