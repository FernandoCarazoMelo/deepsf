class Config():
    """Class settings.
    """

    def __init__(self,
                 epochs=3000,
                 optimizer='adamW',
                 modelNN='DeepSF_2hidden',
                 batch_size=32,
                 learning_rate=1e-1,
                 if_toy=False,
                 if_wandb=False,
                 test_size=0.2,
                 num_genes=898,
                 tumor_type=['LUAD']):
        """Setting deepsf NN characteristics.

        Args:
            batch_size (int, optional): batch size. Defaults to 32.
            learning_rate (float, optional): learning rate. Defaults to 1e-1.
            if_toy (bool, optional): If true you can select the first x num_genes for training; if False just the cancer related genes are used for training. Defaults to True.
            if_wandb (bool, optional): All parameters are tracked by weights and biases. Defaults to False.
            test_size (float, optional): test size. Defaults to 0.2.
            num_genes (int, optional): number of genes included for training. Defaults to 898. If if_toy=False the num_genes is equal to the number of cancer related genes (898). Otherwise you can select the number of genes you want to train the model.
            tumor_type (list, optional): a list of the cancer types used for training. Default is LUAD
        """

        # Screen
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.modelNN = modelNN
        self.epochs = epochs
        self.optimizer = optimizer
        self.num_genes = num_genes
        self.if_toy = if_toy
        self.if_wandb = if_wandb
        self.test_size = test_size
        self.tumor_type = tumor_type

    def get_config(self):
        config = dict(
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            modelNN=self.modelNN,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            if_toy=self.if_toy,
            if_wandb=self.if_wandb,
            test_size=self.test_size,
            num_genes=self.num_genes,
            tumor_type =self.tumor_type
        )
        return config
