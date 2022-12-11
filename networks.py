# Define model
import torch
import torch.nn as nn

class VPT(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 12, 15),
            nn.ReLU(),
            nn.Conv2d(12, 12, 7),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.ReLU()    
        )

        self.depth_stream = nn.Sequential(
            nn.Conv2d(1, 12, 15),
            nn.ReLU(),
            nn.Conv2d(12, 12, 7),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.ReLU()    
        )

        # self.encoder = nn.Sequential(
        #     nn.Conv1d(12, 12, 1),
        #     nn.ELU(),
        # )

        self.rnn = nn.GRU(27, 12, 1)
        # self.rnn = nn.Sequential(
        #     nn.Conv1d(1, 12, 2),
        #     nn.ReLU()
        # )
        # input = torch.randn(12, 9)
        self.h0 = torch.randn(1, 12)
        # output, hn = self.rnn(input, self.h0)

    def forward(self, rgb, odo, depth=None):
        rgb = self.rgb_stream(rgb)
        rgb = torch.flatten(rgb, start_dim=2)
        rgb = torch.sum(rgb, dim=-1)

        depth = self.depth_stream(depth)
        depth = torch.flatten(depth, start_dim=2)
        depth = torch.sum(depth, dim=-1)

        # print(rgb.shape)
        # print(depth.shape)
        # print(odo.shape)

        # fuse
        slab = torch.hstack((rgb, depth, odo))
        # print(slab.shape)

        #rnn
        out, _ = self.rnn(slab, self.h0)

        # out = torch.mean(out, dim=-1)
        # print(out.shape)

        out = out.reshape((3, 4))
        delta_R = out[:, :3]
        delta_t = out[:, -1]
        # delta_R = out[:9].reshape((3, 3))
        # delta_t = out[9:].reshape((3, 1))
        return delta_R, delta_t


# Define model
class VPT_1(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 12, 15),
            nn.ReLU(),
            nn.Conv2d(12, 12, 7),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.ReLU()    
        )

        self.depth_stream = nn.Sequential(
            nn.Conv2d(1, 12, 15),
            nn.ReLU(),
            nn.Conv2d(12, 12, 7),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.ReLU()    
        )

        # self.encoder = nn.Sequential(
        #     nn.Conv1d(12, 12, 1),
        #     nn.ELU(),
        # )

        self.rnn = nn.GRU(27, 12, 1)
        # self.rnn = nn.Sequential(
        #     nn.Conv1d(1, 12, 2),
        #     nn.ReLU()
        # )
        # input = torch.randn(12, 9)
        self.h0 = torch.randn(1, 12)
        # output, hn = self.rnn(input, self.h0)

    def forward(self, rgb, odo, depth=None):
        rgb = self.rgb_stream(rgb)
        rgb = torch.flatten(rgb, start_dim=2)
        rgb = torch.sum(rgb, dim=-1)

        depth = self.depth_stream(depth)
        depth = torch.flatten(depth, start_dim=2)
        depth = torch.sum(depth, dim=-1)

        # print(rgb.shape)
        # print(depth.shape)
        # print(odo.shape)

        # fuse
        slab = torch.hstack((rgb, depth, odo))
        print(slab.shape)

        #rnn
        out, _ = self.rnn(slab, self.h0)

        out = torch.mean(out, dim=-1)
        # print(out.shape)

        out = out.reshape((3, 4))
        delta_R = out[:, :3]
        delta_t = out[:, -1]
        # delta_R = out[:9].reshape((3, 3))
        # delta_t = out[9:].reshape((3, 1))
        return delta_R, delta_t


# Define model
class VPT_RNN(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 12, 15),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 7),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU()   
        )

        self.depth_stream = nn.Sequential(
            nn.Conv2d(1, 12, 15),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 7),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU() 
        )

        # self.encoder = nn.Sequential(
        #     nn.Conv1d(12, 12, 1),
        #     nn.ELU(),
        # )

        # self.rnn = nn.GRU(27, 12, 1)
        # input = [27, 12]
        self.rnn1 = nn.Sequential(
            nn.Conv1d(12, 12, 1),
            # nn.BatchNorm1d(12),
            nn.ReLU()
        )
        self.rnn2 = nn.Sequential(
            nn.Conv1d(12, 12, 1),
            # nn.BatchNorm1d(12),
            nn.ReLU()
        )
        self.rnn3 = nn.Sequential(
            nn.Conv1d(12, 12, 1),
            # nn.BatchNorm1d(12),
            nn.ReLU()
        )
        self.norm1 = nn.BatchNorm1d(27)
        self.norm2 = nn.BatchNorm1d(27)
        self.norm3 = nn.BatchNorm1d(27)
        # input = torch.randn(12, 9)
        self.h0 = torch.randn(1, 12)
        # output, hn = self.rnn(input, self.h0)

    def forward(self, rgb, odo, depth=None):
        rgb = self.rgb_stream(rgb)
        rgb = torch.flatten(rgb, start_dim=2)
        rgb = torch.sum(rgb, dim=-1)

        depth = self.depth_stream(depth)
        depth = torch.flatten(depth, start_dim=2)
        depth = torch.sum(depth, dim=-1)

        # print(rgb.shape)
        # print(depth.shape)
        # print(odo.shape)

        # fuse
        slab = torch.hstack((rgb, depth, odo))
        # print(slab.shape)

        #rnn
        # out, _ = self.rnn(slab, self.h0)
        out = self.rnn1(slab) + slab
        out = self.norm1(out)
        out = self.rnn2(out) + out
        out = self.rnn3(out) + out

        out = torch.mean(out, dim=-1)
        # print(out.shape)

        out = out.reshape((3, 4))
        delta_R = out[:, :3]
        delta_t = out[:, -1]
        # delta_R = out[:9].reshape((3, 3))
        # delta_t = out[9:].reshape((3, 1))
        return delta_R, delta_t


# Define model
class VPT_RNN_2(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 12, 15),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 7),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU()      
        )

        self.depth_stream = nn.Sequential(
            nn.Conv2d(1, 12, 15),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 7),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU()    
        )

        # self.encoder = nn.Sequential(
        #     nn.Conv1d(12, 12, 1),
        #     nn.ELU(),
        # )

        # self.rnn = nn.GRU(27, 12, 1)
        # input = [27, 12]
        self.rnn1 = nn.Sequential(
            nn.Conv1d(12, 12, 1),
            # nn.BatchNorm1d(12),
            nn.ReLU()
        )
        self.rnn2 = nn.Sequential(
            nn.Conv1d(12, 12, 1),
            # nn.BatchNorm1d(12),
            nn.ReLU()
        )
        self.rnn3 = nn.Sequential(
            nn.Conv1d(12, 12, 1),
            # nn.BatchNorm1d(12),
            nn.ReLU()
        )
        self.norm1 = nn.BatchNorm1d(27)
        self.norm2 = nn.BatchNorm1d(27)
        self.norm3 = nn.BatchNorm1d(27)
        # input = torch.randn(12, 9)
        self.h0 = torch.randn(1, 12)
        # output, hn = self.rnn(input, self.h0)

    def forward(self, rgb, odo, depth=None):
        rgb = self.rgb_stream(rgb)
        rgb = torch.flatten(rgb, start_dim=2)
        rgb = torch.sum(rgb, dim=-1)

        depth = self.depth_stream(depth)
        depth = torch.flatten(depth, start_dim=2)
        depth = torch.sum(depth, dim=-1)

        # print(rgb.shape)
        # print(depth.shape)
        # print(odo.shape)

        # fuse
        slab = torch.hstack((rgb, depth, odo))
        # print(slab.shape)

        #rnn
        # out, _ = self.rnn(slab, self.h0)
        out = self.rnn1(slab) + slab
        out = self.norm1(out)
        out = self.rnn2(out) + out
        out = self.norm2(out)
        out = self.rnn3(out) + out
        out = self.norm3(out)

        out = torch.mean(out, dim=-1)
        # print(out.shape)

        out = out.reshape((3, 4))
        delta_R = out[:, :3]
        delta_t = out[:, -1]
        # delta_R = out[:9].reshape((3, 3))
        # delta_t = out[9:].reshape((3, 1))
        return delta_R, delta_t

# Define model
class VPT_RNN_3(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 12, 15),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 7),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU()       
        )

        self.depth_stream = nn.Sequential(
            nn.Conv2d(1, 12, 15),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 7),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU()     
        )

        self.rnn = nn.GRU(27, 12, 1)
        # input = [27, 12]

        self.h0 = torch.randn(1, 12)
        # output, hn = self.rnn(input, self.h0)

    def forward(self, rgb, odo, depth=None):
        rgb = self.rgb_stream(rgb)
        rgb = torch.flatten(rgb, start_dim=2)
        rgb = torch.sum(rgb, dim=-1)

        depth = self.depth_stream(depth)
        depth = torch.flatten(depth, start_dim=2)
        depth = torch.sum(depth, dim=-1)

        # fuse
        slab = torch.hstack((rgb, depth, odo))
        # print(slab.shape)

        #rnn
        self.h0 = self.h0.to(device)
        out, _ = self.rnn(slab, self.h0)


        out = torch.mean(out, dim=-1)
        # print(out.shape)

        out = out.reshape((3, 4))
        delta_R = out[:, :3]
        delta_t = out[:, -1]
        # delta_R = out[:9].reshape((3, 3))
        # delta_t = out[9:].reshape((3, 1))
        return delta_R, delta_t


# Define model
class VPT_RNN_4(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 12, 15),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 7),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU()        
        )

        self.depth_stream = nn.Sequential(
            nn.Conv2d(1, 12, 15),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 7),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3),
            nn.BatchNorm2d(12),
            nn.ReLU()      
        )

        self.rnn = nn.GRU(27, 12, 2)
        # input = [27, 12]

        self.h0 = torch.randn(2, 12)
        # output, hn = self.rnn(input, self.h0)

    def forward(self, rgb, odo, depth=None):
        rgb = self.rgb_stream(rgb)
        rgb = torch.flatten(rgb, start_dim=2)
        rgb = torch.sum(rgb, dim=-1)

        depth = self.depth_stream(depth)
        depth = torch.flatten(depth, start_dim=2)
        depth = torch.sum(depth, dim=-1)

        # fuse
        slab = torch.hstack((rgb, depth, odo))
        # print(slab.shape)

        #rnn
        self.h0 = self.h0.to(device)
        out, _ = self.rnn(slab, self.h0)


        out = torch.mean(out, dim=-1)
        # print(out.shape)

        out = out.reshape((3, 4))
        delta_R = out[:, :3]
        delta_t = out[:, -1]
        # delta_R = out[:9].reshape((3, 3))
        # delta_t = out[9:].reshape((3, 1))
        return delta_R, delta_t