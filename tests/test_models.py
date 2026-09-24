import torch

from cvaegan.losses import class_feature_matching, kl_divergence, mean_feature_matching
from cvaegan.models import CVAEGAN


def test_model_shapes_and_gradients():
    model = CVAEGAN(num_classes=4, latent_dim=32)
    images = torch.randn(2, 3, 128, 128)
    labels = torch.tensor([0, 1])
    latent, mu, logvar = model.encode(images, labels)
    generated = model.generator(latent, labels)
    d_logits, d_features = model.discriminator(generated)
    c_logits, c_features = model.classifier(generated)

    assert latent.shape == (2, 32)
    assert generated.shape == images.shape
    assert d_logits.shape == (2,)
    assert c_logits.shape == (2, 4)
    loss = generated.mean() + d_logits.mean() + c_logits.mean() + kl_divergence(mu, logvar)
    loss.backward()
    assert d_features.ndim == 2 and c_features.ndim == 2


def test_feature_matching_losses_are_finite():
    real = torch.randn(6, 10)
    fake = torch.randn(6, 10, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = mean_feature_matching(real, fake) + class_feature_matching(real, fake, labels, labels)
    assert torch.isfinite(loss)
    loss.backward()

