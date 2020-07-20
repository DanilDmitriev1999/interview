class BERT_biGRU(nn.Module):
    def __init__(self, hidden_dim=256, bidirectional=True, num_layers=1):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
        self.rnn = nn.GRU(768, hidden_dim, bidirectional=bidirectional,
                          num_layers=num_layers, batch_first=True,
                          dropout=0 if num_layers < 2 else 0.1)
        self.drop = nn.Dropout(p=0.1)
        self.linear = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)

    def forward(self, inputs):
        input_ids = inputs[0]
        attention_mask = inputs[1]
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]
        output = self.drop(bert_output)
        _, hidden = self.rnn(output)
        output = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output = self.linear(output).squeeze(1)

        return output