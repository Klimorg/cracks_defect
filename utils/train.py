from model import get_resnet

img_shape = (256, 256, 3)
model = get_resnet(img_shape=img_shape, n_classes=2)
