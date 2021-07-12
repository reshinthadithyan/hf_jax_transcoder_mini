from tree_sitter import Language, Parser
from tree_hugger.core import CPPParser, JavaParser
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

class FunctionExtractor:
  def __init__(self):
    cpp_lang = 'tree-sitter-cpp'
    java_lang = 'tree-sitter-java'

    build = 'build/lang.so'
    Language.build_library(build, [cpp_lang, java_lang])

    self.cp = CPPParser(build)
    self.jp = JavaParser(build)

  def extract_java_methods(self, df):
    data = df.groupby(['language']).get_group(('java'))['content'].dropna().reset_index(drop=True)
    preprocessed_code = []
    
    for code in tqdm(data, total=len(data), desc="Processing java"):
      try:
        same_file_codes = []
        self.jp.parse_code_as_string(code)
        class_names = self.jp.get_all_class_names()
        method_body = self.jp.get_all_method_bodies()
        params = self.jp.get_all_method_names_with_params()

        for class_name in class_names:
          for key, value in method_body[class_name].items():
            complete_code = f"{key} ({', '.join((f'{param[1]} {param[0]}')for param in params[class_name][key])}) {value}"
            same_file_codes.append(complete_code)
        preprocessed_code.append('\n'.join([same_file_code for same_file_code in same_file_codes]))

      except KeyboardInterrupt:
        break

      except AttributeError:
        pass

      except Exception as e:
        print(e)

    final_df = {"code": preprocessed_code}
    final_df = pd.DataFrame(final_df)
    final_df.loc[:, 'language'] = 'java'

    final_df = final_df.replace('', np.NaN).dropna()
    return final_df

  def extract_cpp_functions(self, df):
    data = df.groupby(['language']).get_group(('cpp'))['content'].dropna().reset_index(drop=True)
    preprocessed_code = []

    for code in tqdm(data, total=len(data), desc="Processing cpp"):
      try:
        same_file_codes = []
        self.cp.parse_code_as_string(code)
        func_body = self.cp.get_all_function_bodies()
        params = self.cp.get_all_function_names_with_params()

        for (func, body) in (func_body.items()):
          complete_code = f"{func} ({', '.join((f'{param[1]} {param[0]}') for param in params[func])}) {body}"
          same_file_codes.append(complete_code)
        preprocessed_code.append('\n'.join([same_file_code for same_file_code in same_file_codes]))

      except AttributeError:
        pass

      except KeyboardInterrupt:
        break

      except Exception as e:
        print(e)

    final_df = {"code": preprocessed_code}
    final_df = pd.DataFrame(final_df)
    final_df.loc[:, 'language'] = 'cpp'

    final_df = final_df.replace('', np.NaN).dropna()

    return final_df

  def extract(self):
    raw_data_path = 'raw_data.csv'
    df = pd.read_csv(raw_data_path)
    java_df = self.extract_java_methods(df)
    cpp_df = self.extract_cpp_functions(df)

    final_df = pd.concat([java_df, cpp_df], axis=0)
    final_df.to_csv("processed_data.csv", index=False)
