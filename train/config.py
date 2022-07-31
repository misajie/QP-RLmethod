class Arguments:
    def __init__(self):
        self.gamma = 0.95
        self.hidden_dim = 256
        self.num_layers = 2
        self.num_classes = 1
        self.tau = 0.05
        self.noise_mu = 0
        self.noise_gamma = 0.04
        self.soft_update_freq = 3
        self.epoch = 1000
        self.batch_size = 128
        self.memory_size = 10000
        self.init_episode = 500
        self.buffer_size = 1000
