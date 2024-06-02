import models.u2net as u2net


def load_model(model_name, num_classes):
    if model_name == 'u2net':
        return u2net.u2net_caller(num_classes=num_classes)
    else:
        return None
