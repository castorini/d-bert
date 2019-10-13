import os

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertModel, BertForMaskedLM
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2LMHeadModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F


class BertWrapper(object):

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.cuda()
        self.model.eval()

    def extract_vectors(self, text):
        toks = self.tokenizer.tokenize(text)
        if len(toks) > 512:
            raise ValueError("More than 512 tokens.")
        tokens = ["[CLS]"] + toks + ["[SEP]"]
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(tokens)
        tokens = torch.LongTensor(tokens).unsqueeze(0).cuda()
        segment_ids = torch.LongTensor(segment_ids).unsqueeze(0).cuda()
        input_mask = torch.LongTensor(input_mask).unsqueeze(0).cuda()
        with torch.no_grad():
            embeds = self.model(tokens, segment_ids, input_mask, output_all_encoded_layers=False)[0]
            return toks, embeds[:, 1:-1]

    @classmethod
    def load(cls, model_file):
        cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed_-1")
        tokenizer = BertTokenizer.from_pretrained(model_file, do_lower_case="uncase" in model_file)
        return cls(BertModel.from_pretrained(model_file, cache_dir=cache_dir), tokenizer)


class GPT2PredictionHead(object):

    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model.cuda()
        self.model.eval()

    def predict(self, text, topk=10):
        toks = self.encode(text)
        if len(toks) > 512:
            raise ValueError("More than 512 tokens.")
        tokens = torch.LongTensor(tokens).unsqueeze(0).cuda()
        segment_ids = torch.LongTensor(segment_ids).unsqueeze(0).cuda()
        input_mask = torch.LongTensor(input_mask).unsqueeze(0).cuda()
        period_idx = self.tokenizer.convert_tokens_to_ids(["."])[0]
        semicolon_idx = self.tokenizer.convert_tokens_to_ids([";"])[0]
        with torch.no_grad():
            dist = F.softmax(self.model(tokens, segment_ids, input_mask), 2).data
            _, indices = torch.sort(dist, 2, descending=True)
            dist[:, :, period_idx] = 0
            dist[:, :, semicolon_idx] = 0
            if sample:
                token_indices = Categorical(dist).sample()
            else:
                token_indices = indices[:, :, :10] # dist.max(2)[1]
            # pred_toks = self.tokenizer.convert_ids_to_tokens(token_indices[0, mask_indices].tolist())
            pred_toks = [self.tokenizer.convert_ids_to_tokens(x.tolist()) for x in token_indices[0, mask_indices]]
            return toks, pred_toks


class BertMaskedLMWrapper(object):

    def __init__(self, model, tokenizer, use_parallel=True):
        self.model = model
        if use_parallel:
            self.model = nn.DataParallel(self.model)
        self.tokenizer = tokenizer
        self.model.cuda()
        self.model.eval()

    def predict_text(self, text, sample=False):
        toks = self.tokenizer.tokenize(text)
        if len(toks) > 512:
            raise ValueError("More than 512 tokens.")
        tokens = ["[CLS]"] + toks + [".", "[SEP]"]
        mask_indices = [idx for idx, x in enumerate(tokens) if x == "[UNK]"]
        if len(mask_indices) == 0:
            return None, None
        tokens = ["[MASK]" if x == "[UNK]" else x for x in tokens]
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(tokens)
        tokens = torch.LongTensor(tokens).unsqueeze(0).cuda()
        segment_ids = torch.LongTensor(segment_ids).unsqueeze(0).cuda()
        input_mask = torch.LongTensor(input_mask).unsqueeze(0).cuda()
        period_idx = self.tokenizer.convert_tokens_to_ids(["."])[0]
        semicolon_idx = self.tokenizer.convert_tokens_to_ids([";"])[0]
        with torch.no_grad():
            dist = F.softmax(self.model(tokens, segment_ids, input_mask), 2).data
            _, indices = torch.sort(dist, 2, descending=True)
            dist[:, :, period_idx] = 0
            dist[:, :, semicolon_idx] = 0
            if sample:
                token_indices = Categorical(dist).sample()
            else:
                token_indices = indices[:, :, :10] # dist.max(2)[1]
            # pred_toks = self.tokenizer.convert_ids_to_tokens(token_indices[0, mask_indices].tolist())
            pred_toks = [self.tokenizer.convert_ids_to_tokens(x.tolist()) for x in token_indices[0, mask_indices]]
            return toks, pred_toks

    def iterative_mask_predict(self, text):
        while True:
            _, preds = self.predict_text(text, sample=True)
            if not preds:
                return text.replace(" ##", "")
            text = text.replace("[MASK]", preds[0], 1)

    def iterative_batch_mask_predict(self, texts, single=False):
        fin_texts = []
        continue_texts = texts
        while True:
            if len(continue_texts) == 0:
                return fin_texts
            _, preds_lst = self.batch_predict_text(continue_texts, sample=False)
            next_txts = []
            for preds, txt in zip(preds_lst, continue_texts):
                print(txt)
                if not preds:
                    fin_texts.append(txt.replace(" ##", ""))
                    continue
                next_txts.append(txt.replace("[UNK]", "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" + preds[0], 1))
            continue_texts = next_txts
            if single:
                return continue_texts, fin_texts

    def batch_predict_text(self, texts, sample=False):
        tokens_lst = []
        raw_toks_lst = []
        segment_ids_lst = []
        input_mask_lst = []
        mask_indices_lst = []
        for text in texts:
            toks = self.tokenizer.tokenize(text)
            raw_toks_lst.append(toks)
            if len(toks) > 512:
                raise ValueError("More than 512 tokens.")
            tokens = ["[CLS]"] + toks + [".", "[SEP]"]
            mask_indices = [idx for idx, x in enumerate(tokens) if x == "[UNK]"]
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            tokens = ["[MASK]" if x == "[UNK]" else x for x in tokens]
            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            mask_indices_lst.append(None if len(mask_indices) == 0 else mask_indices)
            tokens_lst.append(tokens)
            segment_ids_lst.append(segment_ids)
            input_mask_lst.append(input_mask)

        max_len = max(len(x) for x in tokens_lst)
        for lst_lst in (tokens_lst, segment_ids_lst, input_mask_lst):
            pad_idx = 1 if lst_lst == input_mask_lst else 0
            for lst in lst_lst: lst.extend([pad_idx] * (max_len - len(lst)))

        tokens = torch.LongTensor(tokens_lst).cuda()
        segment_ids = torch.LongTensor(segment_ids_lst).cuda()
        input_mask = torch.LongTensor(input_mask_lst).cuda()
        period_idx = self.tokenizer.convert_tokens_to_ids(["."])[0]
        semicolon_idx = self.tokenizer.convert_tokens_to_ids([";"])[0]
        # unk_idx = self.tokenizer.convert_tokens_to_ids(["[UNK]"])[0]
        with torch.no_grad():
            dist_batch = F.softmax(self.model(tokens, segment_ids, input_mask), 2).data
            _, indices_batch = torch.sort(dist_batch, 2, descending=True)
            # dist_batch[:] = 0.1
            # for indices, dist in zip(indices_batch.split(1, 0), dist_batch.split(1, 0)):
            #     for idx, slice_ in enumerate(indices.split(1, 1)):
            #         dist[0, idx, slice_.squeeze()[10:]] = 0
            dist_batch[:, :, period_idx] = 0
            dist_batch[:, :, semicolon_idx] = 0
            # dist_batch[:, :, unk_idx] = 0
            if sample:
                token_indices = Categorical(dist_batch).sample()
            else:
                token_indices = dist_batch.max(2)[1]
            pred_toks = []
            for token_indices_, mask_indices in zip(token_indices.split(1, 0), mask_indices_lst):
                if mask_indices is None:
                    pred_toks.append(None)
                else:
                    pred_toks.append(self.tokenizer.convert_ids_to_tokens(token_indices_.squeeze()[mask_indices].tolist()))
            return raw_toks_lst, pred_toks

    @classmethod
    def load(cls, model_fqdn, weights_path=None, **model_kwargs):
        cache_dir = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed_-1")
        tokenizer = BertTokenizer.from_pretrained(model_fqdn, do_lower_case="uncase" in model_fqdn)
        model = BertForMaskedLM.from_pretrained(model_fqdn, cache_dir=cache_dir)
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path), strict=False)
        return cls(model, tokenizer, **model_kwargs)


if __name__ == "__main__":
    lm_model = BertMaskedLMWrapper.load("bert-large-cased")
    import code
    code.interact(local=locals())
    # for _ in range(100):
    #     texts = lm_model.iterative_batch_mask_predict(["i [MASK] [MASK] [MASK] [MASK] in the [MASK]", "the japanese destroyed [MASK] on [MASK]"])
    #     print(texts)