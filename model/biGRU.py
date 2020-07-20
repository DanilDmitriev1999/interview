class GRUBaseline(nn.Module):
    def __init__(self, hidden_dim, emb_dim, n_layers=1, dropout=0.1, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, n_layers, bidirectional=bidirectional)
        self.linear = nn.Linear((1 + bidirectional) * n_layers * hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, seq):
        seq = seq.permute(1, 0)
        emb = self.embedding(seq)
        _, hidden = self.gru(emb)
        hidden = torch.cat([x_0 for x_0 in hidden], -1)
        hidden = self.dropout(hidden)
        output = self.linear(hidden).squeeze(-1)
        output = torch.sigmoid(output)
        return output