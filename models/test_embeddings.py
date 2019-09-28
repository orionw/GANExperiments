import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, XLMTokenizer, XLNetModel, XLNetTokenizer, XLNetLMHeadModel
from xlnet import XLNetForSequenceClassificationGivenEmbedding, XLNetEmbedder, XLNetModelWithoutEmbedding
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)



tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetModel.from_pretrained('xlnet-base-cased')
# lmmodel = XLNetLMHeadModel.from_pretrained('xlnet-base-cased')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1

outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
print(last_hidden_states.shape)

embed_model = XLNetEmbedder.from_pretrained("xlnet-base-cased")
embed_outs = embed_model.encode(input_ids)
last_embedding = embed_outs[0]
print(last_embedding.shape)

assert torch.all(torch.eq(last_embedding, last_hidden_states)), "embeddings were not the same"

# lm_outputs = lmmodel(input_ids)
# last_hidden_states_lm = lm_outputs[0]  # The last hidden-state is the first element of the output tuple
# print(last_hidden_states_lm.shape)

# embed_outs_lm = embed_model(input_ids)
# last_embedding_lm = embed_outs_lm[0]
# print(last_embedding_lm.shape)

# assert torch.all(torch.eq(last_embedding_lm, last_hidden_states_lm)), "LM embeddings were not the same"

discriminator = XLNetForSequenceClassificationGivenEmbedding.from_pretrained('xlnet-base-cased')
output_dis = discriminator(last_embedding)

assert type(output_dis) == torch.Tensor, "output did not work or was not a tensor"

