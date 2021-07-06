
def load_datset_train_tokenization(dataset,tokenizer,tokenizer_trainer,batch_size : int):
    """
    Given Tokenizer(tokenizer.Tokenizer), Dataset(datasets.Dataset) and a Trainer, trains it.

    args:
        dataset           : datasets.Dataset | containing the dataset with key - "code".
        tokenizer         : tokenizers.<TOK>Tokenizer | base tokenizer.
        tokenizer_trainer : tokenizers.<TOK>Trainer | trainer for tokenizers.
        batch_size        : batch_size(int) | batch size to batch the dataset.
    """
    def make_batch_iter(dataset):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["code"]
    tokenizer.train_from_iterator(make_batch_iter(), trainer=tokenizer_trainer, length=len(dataset))
    return tokenizer