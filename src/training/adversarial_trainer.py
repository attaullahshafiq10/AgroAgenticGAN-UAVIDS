import torch
import torch.nn as nn
import torch.optim as optim
from src.models.dual_gan_generator import DualGANGenerator
from src.models.dual_gan_discriminator import DualGANDiscriminator
from configs.hyperparameters import Hyperparameters

def execute_adversarial_training(
    generator: DualGANGenerator,
    discriminator: DualGANDiscriminator,
    dataloader: torch.utils.data.DataLoader,
    hyperparams: Hyperparameters,
    device: torch.device
):
    bce_loss = nn.BCELoss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=hyperparams.generator_lr, betas=(hyperparams.adam_beta1, hyperparams.adam_beta2))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=hyperparams.discriminator_lr, betas=(hyperparams.adam_beta1, hyperparams.adam_beta2))

    for epoch in range(hyperparams.gan_epochs):
        for real_samples in dataloader:
            real_tensors = real_samples[0].to(device)
            current_batch_size = real_tensors.size(0)

            real_labels = torch.ones(current_batch_size, 1, device=device)
            fake_labels = torch.zeros(current_batch_size, 1, device=device)

            discriminator_optimizer.zero_grad()
            real_predictions = discriminator(real_tensors)
            discriminator_real_loss = bce_loss(real_predictions, real_labels)

            noise_tensor = torch.randn(current_batch_size, hyperparams.latent_dim, device=device)
            generated_tensors = generator(noise_tensor)
            fake_predictions = discriminator(generated_tensors.detach())
            discriminator_fake_loss = bce_loss(fake_predictions, fake_labels)

            discriminator_total_loss = discriminator_real_loss + discriminator_fake_loss
            discriminator_total_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad()
            generator_fake_predictions = discriminator(generated_tensors)
            generator_loss = bce_loss(generator_fake_predictions, real_labels)
            generator_loss.backward()
            generator_optimizer.step()
