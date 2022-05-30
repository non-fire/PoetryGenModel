import torch
import _pickle as p

model = torch.load('poetry-gen.pt')
model=model.to(model.device)
max_length = 100
rFile = open('wordDic', 'rb')
vocab = p.load(rFile)

re_vocab = {ix: char for char, ix in list(vocab.items())}

def sample(batch_size,startWord='<START>'):
    poetry = []
    sentence_len = 0
    state = model.begin_state(1)
    output_name = ""
    if (startWord != "<START>"):
        output_name = startWord
        poetry.append(startWord)
    input = torch.Tensor([vocab[startWord]]).view(1, 1).long().to(model.device)
    for i in range(100):
        # 前向计算出概率最大的当前词
        output, state = model(input, state)
        exa=[]
        top_index = output.data[0].topk(3)[1][0].item()
        top2_index = output.data[0].topk(3)[1][1].item()
        top3_index = output.data[0].topk(3)[1][1].item()
        #print(top_index)
        char = re_vocab [top_index]
        if char in poetry and char not in ['。', '！','，']:
            char = re_vocab[top2_index]
        # 遇到终结符则输出
        #if char == '<EOP>':
         #   break
        # 有8个句子则停止预测
        poetry.append(char)
        if char in ['。', '！']:
            sentence_len += 1
            poetry.append('\n')
            if sentence_len == 2:
                #poetry.append(char)
                break
        input = (input.data.new([top_index])).view(1, 1)

    str = ''.join(poetry)
    return str

print(sample(100, "獨".encode('utf-8').decode('UTF-8')))