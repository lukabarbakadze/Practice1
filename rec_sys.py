import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

### read dataset
df = pd.read_csv('Data/train.csv')

### simple preprocessing (assign unique integer to each movie & user & rating)
user_idx = defaultdict(lambda: len(user_idx))
movie_idx = defaultdict(lambda: len(movie_idx))
rating_idx = defaultdict(lambda: len(rating_idx))

df.loc[:, 'user'] = df.user.apply(lambda x: user_idx[x])
df.loc[:, 'movie'] = df.movie.apply(lambda x: movie_idx[x])
df.loc[:, 'rating'] = df.rating.apply(lambda x: rating_idx[x])

# split data
train_df, val_df = train_test_split(df, test_size=0.1)

# define dataset class
class MovieDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        user = self.users[idx]
        movie = self.movies[idx]
        rating = self.ratings[idx]
        return (torch.Tensor(user).long(),
                torch.Tensor(movie).long(),
                torch.Tensor(rating).long())

# build model
class RecommenderSystem(nn.Module):
    def __init__(self, n_users, n_movies, n_embd):
        super().__init__()
        self.user_embd = nn.Embedding(n_users, n_embd)
        self.movie_embd = nn.Embedding(n_movies, n_embd)
        self.ln = nn.Linear(2 * n_embd, 5)
    
    def forward(self, users, movies, ratings=None):
        user_emb = self.user_embd(users)
        movie_emb = self.movie_embd(movies)
        out = torch.cat([user_emb, movie_emb], dim=1)
        out = self.ln(out)
        out = F.softmax(out, dim=-1)
        return out

n_epoch = 10
n_embd = 50
batch_size=5000

train_dataset = MovieDataset(users=torch.Tensor(train_df.user.values).long(),
                             movies=torch.Tensor(train_df.movie.values).long(),
                             ratings=torch.Tensor(train_df.rating.values).long())
val_dataset = MovieDataset(  users=torch.Tensor(val_df.user.values).long(),
                             movies=torch.Tensor(val_df.movie.values).long(),
                             ratings=torch.Tensor(val_df.rating.values).long())

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

model = RecommenderSystem(n_users=df.user.unique().shape[0], 
                          n_movies=df.movie.unique().shape[0], 
                          n_embd=n_embd)

# loss function & optimizer & learning rate scheduler
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=False)

##### training loop
# pbar_epoch = tqdm(position=0, desc='epoch bar', total = n_epoch)
# pbar_batch = tqdm(position=1, desc='Batch bar', total = len(train_dataset), leave=True)
for epoch in range(n_epoch):
    train_epoch_loss = val_epoch_loss = 0
    model.train()
    for n_batch, (user, movie, rating) in enumerate(train_dataloader):
        y_hat = model(user, movie, rating)
        loss = loss_fn(y_hat, rating)
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_epoch_loss += loss.item()

        # pbar_batch.update(1)
        # pbar_batch.set_description(f"loss:{loss.item():.3f}")
        # writer.add_scalar('Loss/train', loss.item(), n_batch)
    model.eval()
    with torch.no_grad():
        for n_batch, (user, movie, rating) in enumerate(val_dataloader):
            y_hat = model(user, movie, rating)
            loss = loss_fn(y_hat, rating)

            val_epoch_loss += loss.item()

    # pbar_epoch.update(1)
    # pbar_batch.reset()
    scheduler.step(train_epoch_loss/len(train_dataloader))
    print(f'Epoch: {epoch+1} | Train Loss: {train_epoch_loss/len(train_dataloader)} | Val Loss: {val_epoch_loss/len(val_dataloader)}')