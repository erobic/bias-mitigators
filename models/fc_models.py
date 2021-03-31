import torch.nn as nn
import torch.nn.functional as F
import torch


class GammaRegressor(nn.Module):
    def __init__(self, in_dims, num_classes, gamma_coeff=5):
        super(GammaRegressor, self).__init__()
        self.fc = nn.Linear(in_dims, num_classes)
        self.gamma_coeff = gamma_coeff

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        # fc = self.gamma_coeff * torch.sigmoid(self.fc(x))
        # fc = torch.sigmoid(self.fc(x))
        fc = F.relu(self.fc(x))
        return {
            'out': fc
        }


class MLP1(nn.Module):
    def __init__(self, in_dims, num_classes, hid_dims=None):
        super(MLP1, self).__init__()
        self.fc = nn.Linear(in_dims, num_classes)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        fc = self.fc(x)
        return {
            'before_logits': x,
            'logits': fc
        }


class MLP2(nn.Module):
    def __init__(self, in_dims, hid_dims, num_classes):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(in_dims, hid_dims)
        self.fc2 = nn.Linear(hid_dims, num_classes)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        x = self.fc1(x)
        x = F.relu(x)
        logits = self.fc2(x)
        return {
            'before_logits': x,
            'logits': logits
        }

    def freeze_layers(self, freeze_layers):
        pass


class MLP3(nn.Module):
    def __init__(self, in_dims, hid_dims, num_classes):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(in_dims, hid_dims)
        self.fc2 = nn.Linear(hid_dims, hid_dims)
        self.fc3 = nn.Linear(hid_dims, num_classes)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        logits = self.fc3(x)
        return {
            'before_logits': x,
            'logits': logits
        }

    def freeze_layers(self, freeze_layers):
        pass


class MLP2_200(MLP2):
    def __init__(self, in_dims=300, hid_dims=600, num_classes=None):
        super().__init__(200, 400, num_classes)


class MLP2_300(MLP2):
    def __init__(self, in_dims=300, hid_dims=600, num_classes=None):
        super().__init__(300, 600, num_classes)


class MLP2_2(MLP2):
    def __init__(self, in_dims=300, hid_dims=600, num_classes=None):
        super().__init__(2, 4, num_classes)


class MLP4(nn.Module):
    def __init__(self, in_dims, hid_dims, num_classes):
        super(MLP4, self).__init__()
        self.fc1 = nn.Linear(in_dims, hid_dims)
        self.fc2 = nn.Linear(hid_dims, hid_dims)
        self.fc3 = nn.Linear(hid_dims, hid_dims)
        self.fc4 = nn.Linear(hid_dims, num_classes)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        logits = self.fc4(x)
        return {
            'before_logits': x,
            'logits': logits
        }


class ModelWrapper(nn.Module):
    def __init__(self, core_model, classifier, classifier_in_layer):
        super().__init__()
        self.core_model = core_model
        self.classifier = classifier
        self.classifier_in_layer = classifier_in_layer

    def forward(self, x):
        out = self.core_model(x)
        feat_repr = out[self.classifier_in_layer]
        return self.classifier(feat_repr)


class LFFMnistClassifier(nn.Module):
    def __init__(self, num_classes=10, in_dims=None, hid_dims=None):
        super(LFFMnistClassifier, self).__init__()
        self.fc1 = nn.Linear(3 * 28 * 28, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)

        self.feature = nn.Sequential(
            nn.Linear(3 * 28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )
        self.classifier = nn.Linear(100, num_classes)

    def forward(self, x, return_feat=False):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        fc1 = x
        x = self.fc2(x)
        x = F.relu(x)
        fc2 = x
        x = self.fc3(x)
        x = F.relu(x)
        fc3 = x
        logits = self.classifier(x)
        return {
            'fc1': fc1,
            'fc2': fc2,
            'fc3': fc3,
            'before_logits': fc3,
            'logits': logits
        }


class MoonNet(nn.Module):
    def __init__(self,
                 hidden_dim=500):
        super(MoonNet, self).__init__()

        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class SlabNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(50, 75, bias=True)
        self.fc2 = nn.Linear(75, 100, bias=True)
        self.classifier = nn.Linear(100, num_classes, bias=True)

    def forward(self, x):
        fc1 = self.fc1(x)
        fc1 = F.relu(fc1)
        before_logits = self.fc2(fc1)
        x = F.relu(before_logits)
        logits = self.classifier(x)
        return {
            'fc1': fc1,
            'fc2': before_logits,
            'before_logits': before_logits,
            'logits': logits
        }

    def forward_representation_encoder(self, x):
        fc1 = self.fc1(x)
        fc1 = F.relu(fc1)
        fc2 = self.fc2(fc1)
        fc2 = F.relu(fc2)
        return {
            'fc1': fc1,
            'fc2': fc2
        }

    def forward_classifier(self, x):
        return self.classifier(x)

    def reset_classifier(self):
        self.classifier.reset_parameters()

    def set_representation_encoder_train(self, is_train):
        self.fc1.train(is_train)
        self.fc2.train(is_train)

    def set_classifier_train(self, is_train):
        self.classifier.train(is_train)

    def get_classifier_named_params(self):
        return (('classifier.weight', self.classifier.weight),
                ('classifier.bias', self.classifier.bias))
