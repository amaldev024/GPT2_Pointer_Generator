# %%
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, GPT2LMHeadModel


from datasets import load_dataset
from transformers import GPT2Tokenizer
import data_loader_gpt_token
from torch.utils.data import Dataset, DataLoader

import wandb


# %%
def get_device():
    #return 'cpu'
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

# %% [markdown]
# Pointer Generator

# %%
class BahdanauAttention(torch.nn.Module):
    
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        #Project to dimension hidden_size
        self.Wa = torch.nn.Linear(hidden_size, hidden_size)
        self.Ua = torch.nn.Linear(hidden_size, hidden_size)
        self.Va = torch.nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

# %%
class Attention(torch.nn.Module):
    """Batch implementation of attentions with Bahdanau attention."""
    
    def __init__(self, emb_dimension=768):
        super(Attention, self).__init__()
        self.add_attention = BahdanauAttention(emb_dimension)

    def forward(self, outputs, mask):

        attentions = torch.zeros(outputs.shape[0], outputs.shape[1], outputs.shape[1]) 
        contexts = torch.zeros(outputs.shape[0], outputs.shape[1], outputs.shape[2])
        
        for i in range(outputs.shape[0]):
            
            start, end = mask[i]
            source = outputs[i:i+1, :start, :]
            summary = outputs[i:i+1, start:, :]
            
            #calculate the attention values for each token in the summary
            for j in range(summary.shape[1]):
                #print(j)
                #print(summary[:, j, :].shape, source.shape)
                con, att = self.add_attention(summary[:, j, :], source)
                #return dimensions are (1, 1, emb_dimension) and (1, 1, seq_len)
                #print(att.squeeze(0).squeeze(0).shape, attentions[i, j, :start].shape)
                attentions[i, j, :start] = att.squeeze(0).squeeze(0)
                contexts[i, j, :] = con.squeeze(0).squeeze(0)
                
        return attentions, contexts
    

# %%
class PointerGenerator(torch.nn.Module):
    """Calculate the pointer generator probability"""
    
    def __init__(self, context_shape=768, hidden_shape=768, input_shape=768):
        super(PointerGenerator, self).__init__()
        self.context = context_shape
        self.dec_hidden_states = hidden_shape
        self.summary_emb = input_shape
        
        self.Wh = torch.nn.Linear(context_shape, 1)
        self.Ws = torch.nn.Linear(hidden_shape, 1)
        self.Wx = torch.nn.Linear(input_shape, 1)
        
    def forward(self,context, dec_hidden_states, summary_emb, mask):
        
        p_gen = torch.zeros(context.shape[0], context.shape[1])
        for i in range(context.shape[0]):
            
            st, en = mask[i]
            
            p_gen[i, st:en] = torch.sigmoid(self.Wh(context[i, st:en, :]) + self.Ws(dec_hidden_states[i, st:en, :]) + self.Wx(summary_emb[i, st:en, :])).squeeze(1)

        
        return p_gen

# %% [markdown]
# Dataset import

# %%


# %%
class CustomDataset(Dataset):
    """Creates a custom dataloade class for dataset fetching"""
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# %% [markdown]
# Model

# %%
def compute_target_mask(x, mask):
    target = (torch.ones(x.shape) * -100).to(get_device())
    for i in range(x.shape[0]):
        st, en = mask[i]
        #copy the target values as target with shifting
        target[i, st:en] = x[i, st+1 : en+1] 
    return target

# %%
class Model(torch.nn.Module):
    def __init__(self, gpt_model, tokenizer, ):
        super().__init__()
        
        self.gpt_model = gpt_model
        self.tokenizer = tokenizer
        self.add_attention = Attention().to(get_device())
        self.pointer_generator = PointerGenerator().to(get_device())
        
    def forward(self, input_tensor, input_data, mask):
        
        token_embeddings = self.gpt_model.transformer.wte(input_tensor.to(get_device()))
        gpt_model_output = self.gpt_model(input_tensor, output_hidden_states=True)
        
        gpt_output_logits = gpt_model_output.logits
        
        gpt_outputs = gpt_model_output.hidden_states[-1]
        
        attentions, context = self.add_attention(outputs=gpt_outputs, mask=mask)
        attentions = attentions.to(get_device())
        context = context.to(get_device())
        
        p_gen = self.pointer_generator(context, gpt_outputs, token_embeddings, mask).to(get_device())
        
        gen_dist = torch.zeros(gpt_model_output.logits.shape)
        
        for i in range(gpt_output_logits.shape[0]):
            st, en = mask[i]
            gen_dist_batch = gpt_output_logits[i, st:en, :]
            gen_dist[i, st:en, :] = F.softmax(gen_dist_batch, dim=-1)
        
        #for handling the out of vocabulary words
        
        batch_unk_tokens = []
        unk_max_len = 0
        #Need to create a new input tensor for the out of vocabulary words else it gives error in backpropagation
        
        
        for i in range(input_tensor.shape[0]):
            #calculate the unk tokens in the input sequence
            new_tokens = data_loader_gpt_token.oov(input_tensor[i], input_data[i], self.tokenizer)
            
            #Find the max length to extend the vocabulary
            unk_max_len = max(unk_max_len, len(new_tokens))
            batch_unk_tokens.append(new_tokens)
        
        
        #Extend the generated distribution by the size of out of vocabulary words
        
        extend_gen_dist = torch.zeros(gen_dist.shape[0], gen_dist.shape[1], unk_max_len)
        gen_dist = torch.cat((gen_dist, extend_gen_dist), dim=2).to(get_device())    
        
        pointer_dist = torch.zeros(gen_dist.shape).to(get_device())
        
        #To construct the pointer distribution 
        
        
        for i,seq in enumerate(input_tensor):
            st, en = mask[i]
            
            
            unk_tokens, new_tokens = data_loader_gpt_token.oov(input_tensor[i], input_data[i], self.tokenizer)
            unk_index = 0
            for j,token in enumerate(seq[:st]):
                
                if token == self.tokenizer.unk_token_id:
                    
                    token = len(self.tokenizer) + new_tokens.index(unk_tokens[unk_index])
                
                pointer_dist[i, st:en+1, token] = attentions[i, st:en+1, j]

        final_distribution = p_gen.unsqueeze(2)*gen_dist + (1-p_gen.unsqueeze(2)) * pointer_dist
        
        return final_distribution
        
        

# %%
def save_checkpoint(model, optimizer, epoch, iteration, filename):

    torch.save({
        'epoch':epoch,
        'iteration': iteration,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict()
    }, filename)

# %%
def train_model(model, optimizer, tokenizer, data_loader, epochs):
    
    loss_per_epoch = []
    min_loss = 1000

    for epoch in range(epochs):
        
        losses = []

        for b_in, batch in enumerate(data_loader):
            
            # Set the model to train mode to enable dropout etc need to change it during evaluation
            model.train()

            input_tensor, input_data = batch[0], batch[1]
            
            num_seg_a = torch.nonzero(input_tensor == tokenizer.encode('<|summarize|>')[0])[:, 1:2] 
            end_index = torch.nonzero(input_tensor == tokenizer.eos_token_id)[:, 1:2]
            mask = torch.cat((num_seg_a, end_index), dim=1).to(get_device())
            
            #pass the input tensor to the model to get the output distribution
            
            final_distribution = model(input_tensor.to(get_device()), input_data, mask)
            
            # Compute the log for calculating the loss
            
            log_final_distribution = torch.log(final_distribution + 1e-8)
            
            target = compute_target_mask(input_tensor, mask)
            
            bat, seq, vocab = final_distribution.shape

            model_output = log_final_distribution.view(bat*seq, vocab)
            target = target.view(bat*seq).to(get_device())
            target = target.long()
            
            loss = F.nll_loss(model_output, target, ignore_index=-100)
            #loss = min(loss, 100)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

            wandb.log({"Batch Loss" : loss.item()})
            
            print(losses)

            if loss.item() < min_loss:
                save_checkpoint(model, optimizer, epoch, b_in, 'model_10ktrain.pth' )

            save_checkpoint(model, optimizer, epoch, b_in, 'model_10ktrain_full.pth' )

    
        epoch_loss = np.average(losses)
        loss_per_epoch.append(epoch_loss)

        wandb.log({"Epoch Loss" : epoch_loss, "Epoch":epoch})
        print('Epoch:', epoch, 'Loss:', epoch_loss)
    wandb.finish()


# %%
#For main class


#importing weightsandbiases
wandb.login(key="5b1542b1afc9b0b7d21909fcf0d8c08e35621bb7")
#initialize new run
wandb.init(project='gpt2-pointet_generator', entity='amaldev04')

#Importing the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2").to(get_device())

#Adding the special tokens for the final model for summariation
special_tokens = {'bos_token':'<|startoftext|>','eos_token':'<|endofthetext|>','pad_token':'<pad>','additional_special_tokens':['<|keyword|>','<|summarize|>']}
tokenizer.add_special_tokens(special_tokens)

#Resizing the token embeddings to include the new tokens 
gpt_model.resize_token_embeddings(len(tokenizer))

traindataset, valdataset = data_loader_gpt_token.data_processor(tokenizer, 10000)
#returns a list of tokenized data as (input_id, type_id, lm_label) 
data_batch = data_loader_gpt_token.smart_batching(traindataset, tokenizer)
#cretes a dataloader object for fetching data
dataset1 = CustomDataset(data_batch)
data_loader = DataLoader(dataset1, batch_size=64, shuffle=True)

model = Model(gpt_model, tokenizer).to(get_device())

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

#To load the model
#checkpoint = torch.load('model_10ktrain.pth')
#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


no_epochs = 10
train_model(model, optimizer, tokenizer, data_loader, no_epochs)
