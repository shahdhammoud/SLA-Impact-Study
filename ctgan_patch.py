from ctgan import CTGAN
import torch

# Patch CTGAN to record generator and discriminator losses per epoch
orig_train = CTGAN._train

def patched_train(self, train_data, train_cond, steps_per_epoch):
    self.generator_losses = []
    self.discriminator_losses = []
    for epoch in range(self.epochs):
        gen_loss_epoch = 0.0
        disc_loss_epoch = 0.0
        for _ in range(steps_per_epoch):
            gen_loss, disc_loss = orig_train(self, train_data, train_cond, 1)
            gen_loss_epoch += gen_loss
            disc_loss_epoch += disc_loss
        self.generator_losses.append(gen_loss_epoch / steps_per_epoch)
        self.discriminator_losses.append(disc_loss_epoch / steps_per_epoch)
    return self

CTGAN._train = patched_train
print('CTGAN patched to record generator and discriminator losses.')

