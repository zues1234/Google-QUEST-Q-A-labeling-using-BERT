class BERTdataset():
    def __init__(self, qtitle, qbody, answer, targets, tokenizer, max_len):
        super(BERTdataset, self).__init__()
        self.qtitle = qtitle
        self.qbody = qbody
        self.answer = answer
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.answer)
    
    def __getitem__(self, idx):
        title = str(self.qtitle[idx])
        body = str(self.qbody[idx])
        answer = str(self.answer[idx])

        input = self.tokenizer.encode_plus(
            f"{title} {body}",
            answer,
            add_special_tokens=True,
            max_len= self.max_len 
        )

        ids = input["input_ids"][0:511]
        mask = input["attention_mask"][0:511]
        token_type_ids = input["token_type_ids"][0:511]

        padding = int(self.max_len - len(ids))
        
        padded_ids = ids + ([0] * padding)
        padded_mask = mask + ([0] * padding)
        padded_token = token_type_ids + ([0] * padding)

        return {
            "ids" : torch.tensor(padded_ids, dtype=torch.long) ,
            "mask" : torch.tensor(padded_mask, dtype=torch.long),
            "token" : torch.tensor(padded_token, dtype=torch.long),
            "targets" : torch.tensor(self.targets[idx,:][0:513], dtype= torch.float)
        }