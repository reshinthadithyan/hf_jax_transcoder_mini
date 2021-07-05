from datasets import load_dataset,Dataset,concatenate_datasets
from pyarrow import LargeStringValue


class CrossLMDataset:
    def __init__(self,dataset_idt="code_x_glue_cc_code_to_code_trans",lang_1_key="java",lang_2_key="cs",bs=8):
        self.train_dataset,self.valid_dataset,self.test_dataset = load_dataset(dataset_idt,split=["train","validation","test"],script_version="master")
        self.lang_1,self.lang_2 = lang_1_key,lang_2_key
        self.bs = bs #batch_size
    def post_process(self,dataset):
        def meta_process(data):
            """additional datapoint level post porcessing"""
            return data
        lang_1_set = Dataset.from_dict({"code":dataset[self.lang_1]})
        lang_2_set = Dataset.from_dict({"code":dataset[self.lang_2]})
        lang_1_set,lang_2_set  = lang_1_set.map(meta_process,batch_size=self.bs),lang_2_set.map(meta_process,batch_size=self.bs)
        mlm_dataset = concatenate_datasets([lang_1_set,lang_2_set])
        return mlm_dataset
    def __call__(self,split="train",preproc_for_crosslm=True):
        if split == "train":
            dataset = self.train_dataset
        elif split == "validation":
            dataset = self.valid_dataset
        elif split == "test":
            dataset = self.test_dataset
        if preproc_for_crosslm:
            dataset = self.post_process(dataset)
        return dataset
if __name__ == "__main__":
    CrossLM = CrossLMDataset()
    print(CrossLM("test"))