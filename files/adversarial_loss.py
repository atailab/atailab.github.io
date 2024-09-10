import math
import torch
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, is_regularize=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_regularize = is_regularize
        self.U = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        self.b = nn.Parameter(torch.Tensor(self.output_dim))

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.input_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, shared_embedding, device):
        loss_l2 = torch.zeros(1, dtype=torch.float32, device=device)
        output = shared_embedding @ self.U + self.b
        if self.is_regularize:
            loss_l2 += torch.norm(self.U)**2/2 + torch.norm(self.b)**2/2

        return output, loss_l2

class Module(nn.Module):
    def __init__(self, user_size, embed_dim, max_seq_len=200, task_num=2, device=torch.device('cuda')):
        '''
        :param user_size: 数据集中的用户个数
        :param embed_dim: embedding的维度
        :param max_seq_len: 级联序列的长度
        '''
        super().__init__()
        self.user_size = user_size
        self.emb_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.task_num = task_num
        self.task_label = torch.LongTensor([i for i in range(self.task_num)])
        self.shared_linear = LinearLayer(self.emb_dim, self.task_num)   # 判别器
        self.user_embedding = nn.Embedding(self.user_size, self.emb_dim)  # 用户的初始特征
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.emb_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def adversarial_loss(self, shared_embedding):
        logits, loss_l2 = self.shared_linear(shared_embedding, self.device)
        loss_adv = torch.zeros(logits.shape[0], device=self.device)
        for task in range(self.task_num):
            label = torch.tensor([task]*logits.shape[0]).to(self.device)
            loss_adv += torch.nn.CrossEntropyLoss(reduce=False)(logits, label.long())

        loss_adv = torch.mean(loss_adv)

        return loss_adv, loss_l2