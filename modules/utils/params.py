comp_coeff = 9  # latent features, number of code
batch_size = 64  # batch size
num_workers = 8  # number of workers for concurrent data loading
max_epochs = 100  # number of epochs
bit_feat = 8  # parameters used for feature encoding.  use default value
alpha = 0.6  # alpha, weight,  values for distillation loss function.