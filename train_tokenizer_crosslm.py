from tokenizer.unified_tokenizer_util import load_datset_train_tokenization
from utils.crosslm_data_utils  import CrossLMDataset
import tokenizers
dataset = CrossLMDataset()("train",False)
def train_tok(dataset,save_dir):
    ''' Initialize and train tokenizer '''
    Tokenizer = tokenizers.ByteLevelBPETokenizer()
    Trainer =  tokenizers.trainers.BpeTrainer(
                vocab_size=3000,
                special_tokens= ["<s>",
                                "<pad>",
                                "</s>",
                                "<unk>",
                                "<mask>"],
                )
    Tokenizer.train_from_iterator(dataset)
    Tokenizer.save_model(save_dir)
if __name__ == "__main__":
    train_tok(dataset,r"/Users/reshinthadithyan/master/research/code-research/unsup_translation/models/crosslm")