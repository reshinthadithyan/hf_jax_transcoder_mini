from datasets import load_dataset,Dataset,concatenate_datasets,DatasetDict

SEED = 42

class CrossLMDataset:
    def __init__(self,dataset_idt="code_x_glue_cc_code_to_code_trans",lang_1_key="java",lang_2_key="cs",bs=8):
        self.train_dataset,self.valid_dataset,self.test_dataset = load_dataset(dataset_idt,split=["train","validation","test"],script_version="master")
        self.lang_1,self.lang_2 = lang_1_key,lang_2_key
        self.bs = bs #batch_size
    def post_process(self,dataset):
        def meta_process(data):
            """additional datapoint level post porcessing"""
            return data
        lang_1_set = Dataset.from_dict({"code":dataset[self.lang_1],"lang":["<j>"]*len(dataset[self.lang_1])})
        lang_2_set = Dataset.from_dict({"code":dataset[self.lang_2],"lang":["<c>"]*len(dataset[self.lang_2])})
        lang_1_set,lang_2_set  = lang_1_set.map(meta_process,batch_size=self.bs),lang_2_set.map(meta_process,batch_size=self.bs)
        mlm_dataset = concatenate_datasets([lang_1_set,lang_2_set]).shuffle(seed=SEED)
        return mlm_dataset
    def __call__(self,split="train",preproc_for_crosslm=True,combine=False):
        """
        Main function to get the CrossLM Dataset as in TransCoder(https://arxiv.org/pdf/2006.03511.pdf).
        Each batch will be made of one language and batches are shuffled.

        args: 
            split (str) - specify the split you want to use.
            preproc_for_crosslm (boolean)- if True, sets preprocesses the model for cross-lingual language modelling 
        """
        if split == "train":
            dataset = self.train_dataset
        elif split == "validation":
            dataset = self.valid_dataset
        elif split == "test":
            dataset = self.test_dataset
        if preproc_for_crosslm:
            if not combine:
                dataset = self.post_process(dataset)
            else:
                trainset,validset,testset = self.post_process(self.train_dataset),self.post_process(self.valid_dataset),self.post_process(self.test_dataset)
                return DatasetDict({"train": trainset,
                        "validation":validset,
                        "test":testset})
        return dataset
if __name__ == "__main__":
    CrossLM = CrossLMDataset()
    print(CrossLM("test",combine=True))
