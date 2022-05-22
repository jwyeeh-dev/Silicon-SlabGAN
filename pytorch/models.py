import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self,opt):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + 1002 +1
        self.input_dim = opt.input_dim # 512+1000+1


        self.l1 = nn.Sequential(nn.Linear(input_dim, 128*1000),nn.ReLU(True))
        self.map1 = nn.Sequential(nn.ConvTranspose2d(128,256,(1,3),stride = 1,padding=0),nn.BatchNorm2d(256,0.8),nn.ReLU(True)) #(28,3)
        self.map2 = nn.Sequential(nn.ConvTranspose2d(256,512,(1,1),stride = 1,padding=0),nn.BatchNorm2d(512,0.8),nn.ReLU(True)) #(28,3)
        self.map3 = nn.Sequential(nn.ConvTranspose2d(512,256,(1,1),stride = 1,padding=0),nn.BatchNorm2d(256,0.8),nn.ReLU(True)) #(28,3)
        self.map4 = nn.Sequential(nn.ConvTranspose2d(256,1,(1,1),stride=1,padding=0)) #(28,3)
        self.cellmap = nn.Sequential(nn.Linear(3000,30),nn.BatchNorm1d(30),nn.ReLU(True),nn.Linear(30,6),nn.Sigmoid())

        self.sigmoid = nn.Sigmoid()

    def forward(self, noise,c1,c2):
        gen_input = torch.cat((noise,c2,c1), -1)
        h = self.l1(gen_input)
        h = h.view(h.shape[0], 128, 1000, 1)
        h = self.map1(h)
        h = self.map2(h)
        h = self.map3(h)
        h = self.map4(h)

        h_flatten = h.view(h.shape[0],-1)
        pos = self.sigmoid(h)
        cell = self.cellmap(h_flatten)
        cell = cell.view(cell.shape[0],1,2,3)
        return torch.cat((cell,pos),dim =2)



class Discriminator(nn.Module):
    def __init__(self,opt):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,3), stride = 1, padding = 0),
                                   nn.LeakyReLU(0.2, inplace=True),nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = (1,1), stride = 1, padding = 0),                                                                             
                                   nn.LeakyReLU(0.2,inplace=True),nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),                                                                               
                                   nn.LeakyReLU(0.2,inplace=True))

        self.avgpool_si = nn.AvgPool2d(kernel_size = (1000,1))

        self.feature_layer = nn.Sequential(nn.Linear(768, 500), nn.LeakyReLU(0.2, inplace =True), nn.Linear(500,200),nn.LeakyReLU(0.2, inplace = True))
        self.output = nn.Sequential(nn.Linear(200,10))

    def forward(self, x):
        B = x.shape[0]
        output = self.model(x)
        output_c = output[:,:,:2,:]
        output_si = output[:,:,2:1002,:]

        output_si = self.avgpool_si(output_si)
        output_all = torch.cat((output_c,output_si),dim=-2)
        output_all = output_all.view(B, -1)
        feature = self.feature_layer(output_all)
        return feature, self.output(feature)


class QHead_(nn.Module):
    def __init__(self,opt):
        super(QHead_,self).__init__()
        self.model_si = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 512, kernel_size = (1,3), stride = 1, padding = 0),
                                      nn.BatchNorm2d(512,0.8),nn.LeakyReLU(0.2,inplace=True),
                                      nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (1,1), stride = 1, padding = 0),
                                      nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),
                                      nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size= (1,1), stride = 1, padding = 0),
                                      nn.BatchNorm2d(256,0.8),nn.LeakyReLU(0.2,inplace=True),                                                                                                                                                     
                                      nn.Conv2d(in_channels = 256, out_channels = 2, kernel_size = (1,1), stride =1, padding =0))

        self.model_cell = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (1,3), stride= 1, padding = 0),
                                        nn.BatchNorm2d(64,0.8),nn.LeakyReLU(0.2,inplace=True),
                                        nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1,1), stride = 1, padding = 0),
                                        nn.BatchNorm2d(64,0.8),nn.LeakyReLU(0.2,inplace=True))

        self.softmax = nn.Softmax2d()
        self.label_si_layer = nn.Sequential(nn.Linear(2000,300),nn.BatchNorm1d(300,0.8),nn.LeakyReLU(0.2,inplace=True),
                                            nn.Linear(300,100),nn.BatchNorm1d(100,0.8),nn.LeakyReLU(0.2,inplace=True),nn.Linear(100,1002),nn.Softmax())
        self.label_c_layer = nn.Sequential(nn.Linear(128,100),nn.BatchNorm1d(100,0.8),nn.LeakyReLU(0.2,inplace=True),nn.Linear(100,50),nn.BatchNorm1d(50,0.8),nn.LeakyReLU(),nn.Linear(50,1),nn.Sigmoid())

    def forward(self, image):
        cell = image[:,:,:2,:]
        si = image[:,:,2:1002,:]

        cell_output = self.model_cell(cell)
        si_output = self.model_si(si)

        #flatten
        cell_output_f = torch.flatten(cell_output,start_dim=1)
        si_output_f = torch.flatten(si_output,start_dim=1)
        si_output_sm = self.softmax(si_output)
        cell_label = self.label_c_layer(cell_output_f)
        si_cat = self.label_si_layer(si_output_f)

        return si_output_sm, si_cat,cell_label

#if __name__ == '__main__':
#   pass



