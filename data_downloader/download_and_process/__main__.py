from download_repos import download_all_repos
from preprocess_data import create_processed_csv
from extract_functions import FunctionExtractor

from typer import run

def main():
    download_all_repos()
    create_processed_csv()
    
    fe = FunctionExtractor()
    fe.extract()
    
if __name__=="__main__":
    run(main)