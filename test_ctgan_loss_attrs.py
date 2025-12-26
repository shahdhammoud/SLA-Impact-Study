from ctgan import CTGAN
import pandas as pd

# Create dummy data
X = pd.DataFrame({
    'A': [0, 1, 0, 1, 0, 1, 0, 1],
    'B': [1, 2, 3, 4, 5, 6, 7, 8]
})

model = CTGAN(epochs=2)
model.fit(X, discrete_columns=['A'])
print('generator_losses:', getattr(model, 'generator_losses', None))
print('discriminator_losses:', getattr(model, 'discriminator_losses', None))

