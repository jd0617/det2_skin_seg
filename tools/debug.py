
import torch
import ultralytics
import torch.nn as nn

from ultralytics import YOLO


class Yolo_Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        import ultralytics
        from ultralytics import YOLO

        yolo = YOLO("yolo11n.pt")

        self.conv1 = yolo.model.model[0]
        self.conv2 = yolo.model.model[1]
        self.c3k2_1 = yolo.model.model[2]
        self.conv3 = yolo.model.model[3]
        self.c3k2_2 = yolo.model.model[4]
        self.conv4 = yolo.model.model[5]
        self.c3k2_3 = yolo.model.model[6]
        self.conv5 = yolo.model.model[7]
        self.c3k2_4 = yolo.model.model[8]

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c3k2_1(x)

        x = self.conv3(x)
        x_8 = self.c3k2_2(x)
        
        x = self.conv4(x_8)
        x_16 = self.c3k2_3(x)
        
        x = self.conv5(x_16)
        x_32 = self.c3k2_4(x)

        return x_8, x_16, x_32


class Yolo_Neck(nn.Module):

    def __init__(self):
        super().__init__()

        import ultralytics
        from ultralytics import YOLO

        yolo = YOLO("yolo11n.pt")

        self.sppf = yolo.model.model[9]
        self.c2psa = yolo.model.model[10]

        self.us1 = yolo.model.model[11]
        self.concat_1 = yolo.model.model[12]
        self.c3k2_1 = yolo.model.model[13]
        self.us2 = yolo.model.model[14]
        self.concat_2 = yolo.model.model[15]
        self.c3k2_2 = yolo.model.model[16]

        self.conv1 = yolo.model.model[17]
        self.concat_3 = yolo.model.model[18]
        self.c3k2_3 = yolo.model.model[19]
        self.conv2 = yolo.model.model[20]
        self.concat_4 = yolo.model.model[21]
        self.c3k2_4 = yolo.model.model[22]

    def forward(self, x_8, x_16, x_32):

        # x_8, x_16, x_32 = x_input

        x_32 = self.sppf(x_32)
        x_32 = self.c2psa(x_32)
        
        x = self.us1(x_32)
        x = self.concat_1([x, x_16])
        x_16 = self.c3k2_1(x)

        x = self.us2(x_16)
        x = self.concat_2([x, x_8])
        x_8 = self.c3k2_2(x)

        x = self.conv1(x_8)
        x = self.concat_3([x, x_16])
        x_16 = self.c3k2_3(x)

        x = self.conv2(x_16)
        x = self.concat_4([x, x_32])
        x_32 = self.c3k2_4(x)

        return x_8, x_16, x_32


class MyYolo(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = Yolo_Backbone()
        self.neck = Yolo_Neck()

        self.det_head = self._get_det_head()

    def _get_det_head(self):
        import ultralytics
        from ultralytics import YOLO

        yolo = YOLO("yolo11n.pt")

        return yolo.model.model[23]

    def forward(self, x):
        x_8, x_16, x_32 = self.backbone(x)

        x_8, x_16, x_32 = self.neck(x_8, x_16, x_32)

        x = self.det_head([x_8, x_16, x_32])

        return x


t = torch.randn([1, 3, 1200, 600])

bb = Yolo_Backbone()
yn = Yolo_Neck()

m = MyYolo()

# byn = nn.Sequential(bb, yn)

# on = byn(t)
o = m(t)

# print(o.shape)

for i in o:
    print(i.shape)

# yolo = YOLO("yolo11n.pt")

# det_head = yolo.model.model[23]

# # 64, 128, 256

# t = [torch.randn([1, 64, 80, 80]), torch.randn([1, 128, 40, 40]), torch.randn([1, 256, 20, 20])]

# o = det_head(t)

# print(o)