class CNN(nn.Module):
    def __init__(self, emb_dim, kernel_sizes, dropout=0.2, n_filters=100):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(ks, emb_dim))
            for ks in kernel_sizes
        ])
        self.linear = nn.Linear(len(kernel_sizes) * n_filters, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.embedding(x).unsqueeze(1)
        conved = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        output = self.linear(cat).squeeze(-1)
        output = torch.sigmoid(output)

        return output