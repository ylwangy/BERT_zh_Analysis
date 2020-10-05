with open('demo.txt','a+') as ff:
	with open('demo.seg','r') as f:
		for line in f:
			if len(line.strip())==0:
				ff.write('\n')
			else:
				if line.strip().split()[1].startswith('B') or line.strip().split()[1].startswith('M'):
					ff.write(line.strip().split()[0])
				else:
					ff.write(line.strip().split()[0])
					ff.write(' ')
		ff.write('\n')

with open('demo_len.txt','a+') as ff:
	with open('demo.txt','r') as f:
		for line in f:
			if len(line.strip())==0:
				continue
			ff.write(line.strip())
			ff.write('|||')
			for i,s in enumerate(line.strip().split()):
				ff.write(str(len(s)))
				ff.write(' ')
			ff.write('\n')


ids=[]

with open ('demo_len_pos.txt','a+') as ff:
	with open ('demo_len.txt','r') as f:
		for line in f:
			ids=[]
			
			if len(line.strip())==0:
				ff.write('\n')
				continue
			l=line.strip().split('|||')[0].strip()
			print(l)
			for c in range(len(l)):
				if c==0:
					ids.append(c)
				else:
					if l[c]!=' ' and l[c-1]== ' ':
						ids.append(c)
					elif l[c]==' ':
						ids.append('s')
					else :
						ids.append('x')

			print(ids)
			ss=[]
			numofs=0
			for i in ids:
				if i is not 's' and i is not 'x':
					ss.append(i-numofs)
				elif i is 'x':
					ss.append(i)
				else:
					numofs+=1
			print(ss)

			sss=[]
			for i in range(len(ss)):
				if ss[i]=='x':
					sss.append('x')
					continue
				numofx=0
				for ii in range(i+1,len(ss)):
					if ss[ii]=='x':
						numofx+=1
					else:
						break
				sss.append(numofx)
			assert len(ss)==len(sss)


			print(sss)
			final=[]
			for i in range(len(ss)):
				if ss[i]=='x':
					final.append('x')
				else:
					final.append(str(ss[i]+sss[i]))

			print(final)
			ff.write(line.strip())
			ff.write(' ||| ')
			ff.write(line.strip().split('|||')[0].strip().replace(' ',''))
			ff.write(' ||| ')
			ff.write(' '.join(final))
			ff.write('\n')


import torch
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese',output_hidden_states=True,output_attentions=True)
model.eval()
model.to('cuda')


d1=0
d2=0
d3=0
d4=0
with open('demo_filter.txt','a+') as ff:
	with open('demo_len_pos.txt','r') as f:
		c_=0
		t_=0
		for line in f:
			if len(line.strip()) ==0:
				continue
			if not len(line.split('|||')) ==4:
				d1+=1
				continue
			sent = line.split('|||')[2].strip()#改革开放三十多年了。
			label = line.split('|||')[3].strip()#1 x 3 x 6 x x 7 8 9
			if not len(sent) == len(label.split(' ')):
				print(line)
				d2+=1
				continue

			if len(sent) >100:
				d3+=1
				continue

			sent_ = '[CLS]' + sent + '[SEP]'
			# print(sent_)
			str_tokenized_sents = tokenizer.tokenize(sent_)
			indexed_tokens = tokenizer.convert_tokens_to_ids(str_tokenized_sents)
			tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')

			with torch.no_grad():
				outputs = model(tokens_tensor)
				att= outputs[3]

				attmap = att[0][:,0,:,:]
			if len(label.split(' ')) + 2 == attmap.size(-1):
				ff.write(line.strip())
				ff.write('\n')
			else:
				d4+=1

print(d1)
print(d2)
print(d3)
print(d4)

with open ('demo_final.txt','a+') as ff:
	with open ('demo_filter.txt','r') as f:
		for line in f:
			if len(line.strip())==0:
				continue
			s=line.strip().split('|||')[2].strip()
			l=line.strip().split('|||')[3].strip()
			l=l.split()
			if len(l)<3:
				continue
			print(l)

			t_h=[]
			tmp=0
			for i in range(len(l)):
				if not l[i]=='x':
					tmp=str(i)
					if tmp==l[i]:
						t_h.append(tmp)
					else:
						t_h.append('x')
				elif i ==len(l)-1:
					t_h.append(tmp)
				elif l[i]=='x' and l[i+1] !='x' :
					t_h.append(tmp)
				else:
					t_h.append('x')
			print(t_h)

			h_h=[]
			for i in range(len(l)):
				if not l[i]=='x':
					if int(l[i])<len(l)-1:
						h_h.append(str(int(l[i])+1))
					else:
						h_h.append('x')
				else:
					h_h.append('x')
			print(h_h)
			t_t=[]
			for i in range(len(t_h)):
				if not t_h[i]=='x':
					if int(t_h[i])>0:
						t_t.append(str(int(t_h[i])-1))
					else:
						t_t.append('x')
				else:
					t_t.append('x')
			print(t_t)

			ff.write(s)
			ff.write('|||')
			ff.write(' '.join(l))
			ff.write('|||')
			ff.write(' '.join(t_h))
			ff.write('|||')
			ff.write(' '.join(h_h))
			ff.write('|||')
			ff.write(' '.join(t_t))
			ff.write('\n')