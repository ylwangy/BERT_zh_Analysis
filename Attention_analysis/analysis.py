#attention distribution for head->head, head->tail, tail->head and tail->tail patterns
import torch
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese',output_hidden_states=True,output_attentions=True)
model.eval()
model.to('cuda')

layers_num=0
heads_num=0


def evaluate(attmap,l, h_t,t_h,h_h,t_t):

	attwithoutclssep = attmap.squeeze(0)
	h_t_alpha=0
	t_h_alpha=0
	h_h_alpha=0
	t_t_alpha=0
	h_t_n=0
	t_h_n=0
	h_h_n=0
	t_t_n=0
	nwords= l-2
	for i in range(1,l-1):
		s = attwithoutclssep[i][1:-1].sum()
		if not h_t[i-1] == 'x' and (not int(h_t[i-1])==i) and (not int(h_t[i-1])==i-1):
			h_t_alpha+=attwithoutclssep[i][int(h_t[i-1])+1]/s
			h_t_n+=1
		if not t_h[i-1] == 'x' and (not int(t_h[i-1])==i-2) and (not int(t_h[i-1])==i-1):
			t_h_alpha+=attwithoutclssep[i][int(t_h[i-1])+1]/s
			t_h_n+=1
		if not h_h[i-1] == 'x' and (not int(h_h[i-1])==i+1) and (not int(h_h[i-1])==i):
			h_h_alpha+=attwithoutclssep[i][int(h_h[i-1])+1]/s
			h_h_n+=1
		if not t_t[i-1] == 'x' and (not int(t_t[i-1])==i-3) and (not int(t_t[i-1])==i-2):
			t_t_alpha+=attwithoutclssep[i][int(t_t[i-1])+1]/s
			t_t_n+=1
	return 	h_t_alpha,t_h_alpha,h_h_alpha,t_t_alpha,h_t_n,t_h_n,h_h_n,t_t_n
for layers_num in [0,1,2,3,4,5,6,7,8,9,10,11]:#[0,1,2,3,4,5,6,7,8,9,10,11]
	for heads_num in [0,1,2,3,4,5,6,7,8,9,10,11]:
		with open('demo_final.txt','r') as f:
			v1=0
			v2=0
			v3=0
			v4=0

			n1=0
			n2=0
			n3=0
			n4=0
			for line in f:
				sent = line.split('|||')[0].strip()#
				h_t = line.split('|||')[1].strip().split()#
				t_h = line.split('|||')[2].strip().split()#
				h_h = line.split('|||')[3].strip().split()#
				t_t = line.split('|||')[4].strip().split()#
				sent_ = '[CLS]' + sent + '[SEP]'
				str_tokenized_sents = tokenizer.tokenize(sent_)
				indexed_tokens = tokenizer.convert_tokens_to_ids(str_tokenized_sents)
				tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')

				with torch.no_grad():
					outputs = model(tokens_tensor)
					# print(len(outputs))
					encoded_layers = outputs[0]
					hidden_states=outputs[2]
					att= outputs[3]
					attmap = att[layers_num][:,heads_num,:,:]

				h_t_alpha,t_h_alpha,h_h_alpha,t_t_alpha,h_t_n,t_h_n,h_h_n,t_t_n = evaluate(attmap,len(sent)+2,h_t,t_h,h_h,t_t)
				v1+=h_t_alpha
				v2+=t_h_alpha
				v3+=h_h_alpha
				v4+=t_t_alpha
				n1+=h_t_n
				n2+=t_h_n
				n3+=h_h_n
				n4+=t_t_n
		
		print('the %d layers %d heads, h_t, t_h, h_h, t_t  is %f, %f, %f, %f'%(int(layers_num), int(heads_num), v1/n1, v2/n2, v3/n3, v4/n4))


