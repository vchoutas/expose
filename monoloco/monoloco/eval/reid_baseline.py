
import torch
import torch.backends.cudnn as cudnn
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


from ..utils import open_image


def get_reid_features(reid_net, boxes, boxes_r, path_image, path_image_r):

    pil_image = open_image(path_image)
    pil_image_r = open_image(path_image_r)
    assert boxes and boxes_r
    cropped_img = []
    for box in boxes:
        cropped_img = cropped_img + [pil_image.crop((box[0], box[1], box[2], box[3]))]
    cropped_img_r = []
    for box in boxes_r:
        cropped_img_r = cropped_img_r + [pil_image_r.crop((box[0], box[1], box[2], box[3]))]

    features = reid_net.forward(cropped_img)
    features_r = reid_net.forward(cropped_img_r)
    return features.cpu(), features_r.cpu()


class ReID(object):
    def __init__(self, weights_path, device, num_classes=751, height=256, width=128):
        super(ReID, self).__init__()
        torch.manual_seed(1)
        self.device = device

        if self.device.type == "cuda":
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(1)
        else:
            print("Currently using CPU (GPU is highly recommended)")

        self.transform_test = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("ReID Baseline:")
        print("Initializing ResNet model")
        self.model = ResNet50(num_classes=num_classes, loss={'xent'})
        self.model.to(device)
        num_param = sum(p.numel() for p in self.model.parameters()) / 1e+06
        print("Model size: {:.3f} M".format(num_param))

        # load pretrained weights but ignore layers that don't match in size
        checkpoint = torch.load(weights_path)
        model_dict = self.model.state_dict()
        pretrain_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        self.model.load_state_dict(model_dict)
        print("Loaded pretrained weights from '{}'".format(weights_path))
        self.model.eval()

    def forward(self, images):
        image = torch.stack([self.transform_test(image) for image in images], dim=0)

        image = image.to(self.device)
        with torch.no_grad():
            features = self.model(image)
        return features

    @staticmethod
    def calculate_distmat(features_1, features_2=None, use_cosine=False):
        query = features_1
        if features_2 is not None:
            gallery = features_2
        else:
            gallery = features_1
        m = query.size(0)
        n = gallery.size(0)
        if not use_cosine:
            distmat = torch.pow(query, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(gallery, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            distmat.addmm_(1, -2, query, gallery.t())
        else:
            features_norm = query/query.norm(dim=1)[:, None]
            reference_norm = gallery/gallery.norm(dim=1)[:, None]
            distmat = torch.mm(features_norm, reference_norm.transpose(0, 1))
        return distmat


class ResNet50(nn.Module):
    def __init__(self, num_classes, loss):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        return y, f
