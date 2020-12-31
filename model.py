class BERTModel(nn.Module):
    def __init__(self, bert_path):
        super(BERTModel, self).__init__()

        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 30)

    def forward(self, ids, attention_mask , token_type_ids):
        _, output2 = self.bert(ids, attention_mask, token_type_ids)
        dropout = self.bert_drop(output2)
        output = self.out(dropout)
        return output 