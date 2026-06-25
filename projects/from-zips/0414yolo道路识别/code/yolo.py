class SegMaskPSP(nn.Module):  # PSP头，多了RFB2和FFM，同样砍了通道数，没找到合适的位置加辅助损失，因此放弃辅助损失
    def __init__(self, n_segcls=19, n=1, c_hid=256, shortcut=False, ch=()):  # n是C3的, (接口保留了,没有使用)c_hid是隐藏层输出通道数（注意配置文件s*0.5,m*0.75,l*1）
        super(SegMaskPSP, self).__init__()
        self.c_in8 = ch[0]  # 16  # 用16,19,22宁可在融合处加深耗费一些时间，检测会涨点分割也很好。严格的消融实验证明用17,20,23分割可能还会微涨，但检测会掉３个点以上，所有头如此
        self.c_in16 = ch[1]  # 19
        self.c_in32 = ch[2]  # 22
        # self.c_aux = ch[0]  # 辅助损失  找不到合适地方放辅助，放弃
        self.c_out = n_segcls
        # 注意配置文件通道写256,此时s模型c_hid＝128
        self.out = nn.Sequential(  # 实验表明引入较浅非线性不太强的层做分割会退化成检测的辅助(分割会相对低如72退到70,71，检测会明显升高)，PP前应加入非线性强一点的层并适当扩大感受野
                                RFB2(c_hid*3, c_hid, d=[2,3], map_reduce=6),  # 3*128//6=64　RFB2和RFB无关，仅仅是历史遗留命名(训完与训练模型效果不错就没有改名重训了)
                                PyramidPooling(c_hid, k=[1, 2, 3, 6]),  # 按原文1,2,3,6，PSP加全局更好，但是ASPP加了全局后出现边界破碎
                                FFM(c_hid*2, c_hid, k=3, is_cat=False),  # FFM改用k=3, 相应的砍掉部分通道降低计算量(原则就是差距大的融合哪怕砍通道第一层也最好用3*3卷积，FFM融合效果又比一般卷积好，除base头外其他头都遵循这种融合方式)
                                nn.Conv2d(c_hid, self.c_out, kernel_size=1, padding=0),
                                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
                               )
        self.m8 = nn.Sequential(
                                Conv(self.c_in8, c_hid, k=1),
        )
        self.m32 = nn.Sequential(
                                Conv(self.c_in32, c_hid, k=1),
                                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        )
        self.m16 = nn.Sequential(
                                Conv(self.c_in16, c_hid, k=1),
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        # self.aux = nn.Sequential(
        #                        Conv(self.c_aux, 256, 3),
        #                        nn.Dropout(0.1, False),
        #                        nn.Conv2d(256, self.c_out, kernel_size=1),
        #                        nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        # )
    def forward(self, x):
        # 这个头三层融合输入做过消融实验，单独16:72.6三层融合:73.5,建议所有用1/8的头都采用三层融合，在Lab的实验显示三层融合的1/16输入也有增长
        feat = torch.cat([self.m8(x[0]), self.m16(x[1]), self.m32(x[2])], 1)
        # return self.out(feat) if not self.training else [self.out(feat), self.aux(x[0])]
        return self.out(feat)
