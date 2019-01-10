import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch_geometric
import torch_geometric.transforms as T
from numpy import random
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv

import _pickle as pickle
from fetcher import DataFetcher

EXP = "VAE-V3"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########   MODEL   #########


class Pool(nn.Module):
    def __init__(self, M, N, I, K):  # M=num_points_pre N=num_points_post K=output_dim
        super(Pool, self).__init__()
        self.conv1 = nn.Conv1d(I, M, 1)
        self.norm1 = nn.InstanceNorm1d(N)
        self.conv2 = nn.Conv1d(M, N, 1)
        self.norm2 = nn.InstanceNorm1d(N)
        self.upsample = nn.Conv1d(1, K, 1)

    def forward(self, data, edge_index):
        x = data.pos.view(1, data.pos.size(0), data.pos.size(1))  # (1, M, I)
        x = x.transpose(2, 1)  # (1, I, M)
        M = x.size(2)
        x = F.relu(self.norm1(self.conv1(x)))  # (1, N, M)
        x = F.relu(self.norm2(self.conv2(x)))  # (1, N, M)
        x = F.max_pool1d(x, M, 1)  # (1, N, 1)
        x = x.transpose(2, 1)  # (1, 1, N)
        x = F.relu(self.upsample(x))  # (1, K, N)
        data.pos = x.transpose(2, 1).squeeze(0)  # (N, K)
        data.edge_index = edge_index
        return data


class STN(nn.Module):
    def __init__(self, num_points=51*51, dim=3):
        super(STN, self).__init__()
        self.K = dim
        self.N = num_points
        self.identity = torch.eye(self.K).view(-1).to(device)
        self.block1 = nn.Sequential(
            nn.Conv1d(self.K, 64, 1),
            nn.InstanceNorm1d(64),
            nn.ReLU())
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.InstanceNorm1d(128),
            nn.ReLU())
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.InstanceNorm1d(1024),
            nn.ReLU())
        self.fc1 = nn.Linear(1024, 512)
        self.norm1 = nn.InstanceNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.norm2 = nn.InstanceNorm1d(256)
        self.fc3 = nn.Linear(256, self.K * self.K)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.max_pool1d(x, self.N).squeeze(2)
        x = self.norm1(self.fc1(x).unsqueeze(0))
        x = F.relu(x).squeeze(0)
        x = self.norm2(self.fc2(x).unsqueeze(0))
        x = F.relu(x).squeeze(0)
        x = self.fc3(x)
        x += self.identity
        x = x.view(-1, self.K, self.K)
        return x


class PointNet(nn.Module):
    def __init__(self, num_points=51*51, K=3):
        super(PointNet, self).__init__()
        hidden_size = config.hidden_size
        self.input_transformer = STN(num_points, K)
        self.embedding_transformer = STN(num_points, 64)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(K, 64, 1),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.InstanceNorm1d(64),
            nn.ReLU())
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.InstanceNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.InstanceNorm1d(1024),
            nn.ReLU())
        self.fc1 = nn.Linear(1024, hidden_size)
        self.fc2 = nn.Linear(1024, hidden_size)

    def forward(self, x):
        N = x.size(2)
        trans = self.input_transformer(x)
        x = torch.bmm(trans, x)
        x = self.mlp1(x)
        trans = self.embedding_transformer(x)
        x = torch.bmm(trans, x)
        x = self.mlp2(x)
        x = F.max_pool1d(x, N).squeeze(2)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        return mu, logvar


class Block(nn.Module):
    def __init__(self, num_channels):
        super(Block, self).__init__()
        self.conv1 = ChebConv(num_channels, num_channels, 6)
        self.conv2 = ChebConv(num_channels, num_channels, 6)
        self.conv3 = ChebConv(num_channels, num_channels, 6)

    def forward(self, data):
        residual = data.pos
        data.pos = F.relu(self.conv1(data.pos, data.edge_index))
        data.pos = F.relu(self.conv2(data.pos, data.edge_index))
        data.pos = F.relu(self.conv3(data.pos, data.edge_index))
        data.pos += residual
        return data

# VAE Decoder Network


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        num_channels = config.channel_conv
        self.num_channels = num_channels
        hidden_size = config.hidden_size

        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 51*51*6)

        self.conv_init = ChebConv(9, num_channels, 6)

        block_seq = []
        for _ in range(0, 6):
            block_seq += [Block(num_channels)]
        self.blocks = nn.Sequential(*block_seq)

        self.conv_final = ChebConv(num_channels, 3, 6)

    def forward(self, vector, edges):
        vector = F.relu(self.fc1(vector))
        vector = F.relu(self.fc2(vector))
        vector = vector.view(-1, 6)
        sphere_pos = sphere.pos
        vector = torch.cat([sphere_pos, vector], dim=1)
        vector = self.conv_init(vector, edges)
        data = Data(pos=vector, edge_index=edges)
        data = self.blocks(data)
        data.pos = self.conv_final(data.pos, data.edge_index)
        data.pos = torch.tanh(data.pos)
        return data

# VAE Model


class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        self.encoder = PointNet()
        self.decoder = Decoder(config)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.tensor(std.data.new(std.size()).normal_()).to(device)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, data, edges):
        x = data.pos.view(1, data.pos.size(0), data.pos.size(1))
        x = x.transpose(2, 1)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        data = self.decoder(z, edges)
        return data, mu, logvar


def chamfer_distance(p1, p2):
    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)
    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)
    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)
    dist, indice = torch.min(dist, dim=2)
    return dist.sum(), indice


def chamfer_loss(x, y):
    chamfer1, _ = chamfer_distance(x, y)
    chamfer2, _ = chamfer_distance(y, x)
    return torch.mean(chamfer1 + chamfer2*0.55)

def mesh_loss(vertice, target_data, edges, normal_loss=False):
    target, gt_norm = target_data[:, 0:3], target_data[:, 3:6]

    # Edge Loss
    pos1 = torch.index_select(target, 0, edges[0])
    pos2 = torch.index_select(vertice, 0, edges[1])
    edge_dis = pos1 - pos2
    edge_length = torch.sum(torch.mul(edge_dis, edge_dis), dim=1)
    edge_loss = torch.mean(edge_length)

    # Chamfer Loss 
    chamfer1, _ = chamfer_distance(vertice, target)
    chamfer2, indice2 = chamfer_distance(target, vertice)
    vertice_loss = chamfer1 + chamfer2 * 0.55

    # Norm Loss
    if not normal_loss:
        return edge_loss + vertice_loss
    else:
        norm = torch.index_select(gt_norm, 0, torch.squeeze(indice2, 0))
        norm = torch.index_select(norm, 0, edges[0])
        norm = torch.mul(norm.norm(p=2, dim=1), edge_dis.norm(p=2, dim=1))
        cosine = torch.abs(torch.sum(norm))
        norm_loss = torch.mean(cosine) * 0.001
        return edge_loss + vertice_loss + norm_loss

def kld_loss(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

########   TRAIN   #########


random.seed(int(time.time()))


def print_network(net, name="Neural Network"):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('-- %s Number of Parameters: %d' % (name, num_params))
    print('-- %s Architecture' % name)
    print(net)


def save_obj(points, faces, file="test.obj"):
    obj = open(file, 'w+')
    for item in points:
        item = item
        obj.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))
    for item in faces:
        obj.write(
            "f {0} {1} {2}\n".format(item[0], item[1], item[2]))
    obj.close()


def sample(network, input, vae_mode=False, name="test.obj"):
    edge_index = sphere.edge_index.to(device) - 1
    input = torch.tensor(input)
    if not vae_mode:
        mesh = network.decoder(input, edge_index)
    else:
        input = Data(pos=input.to(device), edge_index=edge_index)
        mesh, _, _ = network(input, edge_index)
    points = mesh.pos
    points = torch.mul(points, 1.25)
    points = points.cpu().detach().numpy()
    save_obj(points, sphere.face.transpose(1, 0).cpu().numpy(), file=name)


class config(object):
    channel_conv = 128
    num_vertice = 51*51
    hidden_size = 32


if __name__ == '__main__':

    global sphere
    sphere = pickle.load(open("sphere.pkl", "rb"))
    edges = sphere.edge_index.to(device) - 1
    pos = sphere.pos.to(device)
    pos = torch.mul(pos, 0.8)
    sphere.pos = pos

    VAE = VAE(config).to(device)

    VAE.train()
    VAE.encoder.train()
    VAE.decoder.train()

    print_network(VAE.encoder, "PointNet Encoder")
    print_network(VAE.decoder, "ChebConv Decoder")

    optim_VAE = optim.Adam(VAE.parameters(), lr=1e-5, betas=(0.5, 0.5))

    L2 = nn.MSELoss()

    dataset = DataFetcher("train_list.txt")
    dataset.start()
    timer = time.time()
    for step in range(10000000000):
        _, point, _ = dataset.fetch()
        if point.shape[0] < 51*51:
            step -= 1
            continue
        if point.shape[0] > 51*51:
            index = point.shape[0]
            while index > 51*51:
                rand_int = random.randint(0, index - 2)
                point[rand_int, :] = point[index - 1, :]
                index = index - 1
            point = point[0:51*51][:]

        point_with_norm = point
        point_with_norm = torch.tensor(point_with_norm).to(device)

        point = point[:, 0:3]
        point = torch.tensor(point).to(device)
        point = torch.mul(point, 0.8)

        point_with_norm[:, 0:3] = point

        rec, mu, logvar = VAE(Data(pos=point, edge_index=edges), edges)

        shift_loss = L2(rec.pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0) * 10
        reconstruction_loss = mesh_loss(rec.pos, point_with_norm, edges, False) * 0.005
        prior_loss = kld_loss(mu, logvar) * 0.01

        vae_loss = prior_loss + reconstruction_loss + shift_loss

        optim_VAE.zero_grad()
        vae_loss.backward()
        optim_VAE.step()

        if (step + 1) % 5 == 0:
            string = "Step %d - Chamfer:%.3f Shift:%.3f KLD:%.3f Total:%.3f, Speed: %.2f sec/step"
            print(string % (step+1, reconstruction_loss, shift_loss,
                            prior_loss, vae_loss, (time.time()-timer)/5))
            timer = time.time()

        if (step + 1) % 50 == 0:
            print("Ori :\n", point_with_norm[:, 0:3])
            print("Rec :\n", rec.pos)

        if (step + 1) % 500 == 0:
            print("Saving models and samples")

            if not os.path.isdir("checkpoints/%s" % EXP):
                os.makedirs("checkpoints/%s" % EXP)

            status = {'VAE': VAE.state_dict(), 'step': step}
            torch.save(status, "checkpoints/%s/%d.nn" % (EXP, step))

            fixed_noise = torch.randn(1, config.hidden_size).to(device) - 0.5

            sample(VAE, fixed_noise, vae_mode=False,
                   name="checkpoints/%s/%d.obj" % (EXP, step))
            sample(VAE, sphere.pos, vae_mode=True,
                   name="checkpoints/%s/sphere_%d.obj" % (EXP, step))

    dataset.stopped = True
