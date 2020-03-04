import os

from data_loader import *
from model import *


class Executor:
    def __init__(self, prices_csv="BTC-ETH-filtered_with_indicators.csv", batch_size=32, seq_size=24):
        # Data load
        self.dataset = CryptoDataset(csv_file=prices_csv, predict_delta=1, batch_size=batch_size, sequence_size=seq_size)
        # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
        self.device = self.get_available_device()
        self.features_count = self.dataset.train_data.shape[1]
        # Prepare the LSTM model
        self.model = PriceLSTM(features_count=self.features_count, device=self.device, sequence_size=seq_size)
        self.model.to(self.device)  # push model to device in order to have both model and variables on the same execution device

    @staticmethod
    def get_available_device():
        # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def train(self):
        lr = 0.0003
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        epochs = 300
        counter = 0
        print_every = 5
        clip = 5
        valid_loss_min = np.Inf
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                           steps_per_epoch=len(self.dataset.train_loader),
                                                           epochs=epochs)
        train_losses, valid_losses = [], []

        def train():
            self.model.train()
            for train, train_target in self.dataset.train_loader:
                if train.shape[0] < 2:  # No need to skip this since have dropped the bn
                    continue
                # Load data as a torch tensor with gradient accumulation abilities
                train = train.requires_grad_().to(self.device)
                train_target = train_target.to(self.device)
                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()
                # Forward pass to get output/logits
                outputs = self.model(train)
                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs, train_target)
                # Getting gradients w.r.t. parameters
                loss.backward()
                # Updating parameters
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                optimizer.step()
                lr_scheduler.step()

        def eval_and_save(on_epoch=0):
            self.model.eval()
            global valid_loss_min

            # evaluate first the full train set
            epoch_train_losses = []
            for train, train_target in self.dataset.train_loader:
                # Forward pass to get output/logits
                train = train.to(self.device)
                train_target = train_target.to(self.device)
                outputs = self.model(train)
                train_loss = criterion(outputs, train_target)
                epoch_train_losses.append(train_loss.item())
            avg_epoch_train_loss = np.mean(epoch_train_losses)
            train_losses.append(avg_epoch_train_loss)

            # eval the valid set
            epoch_val_losses = []
            for val_in, val_out in self.dataset.val_loader:
                # Forward pass to get output/logits
                val_in = val_in.to(self.device)
                val_out = val_out.to(self.device)
                outputs = self.model(val_in)
                val_loss = criterion(outputs, val_out)
                epoch_val_losses.append(val_loss.item())
            avg_epoch_val_loss = np.mean(epoch_val_losses)
            valid_losses.append(avg_epoch_val_loss)

            print("Epoch: {}...".format(epoch),
                  "Loss: {:.6f}...".format(avg_epoch_train_loss),
                  "Val Loss: {:.6f}".format(avg_epoch_val_loss))
            if avg_epoch_val_loss < valid_loss_min:
                torch.save(self.model.state_dict(), f'{os.path.splitext(prices_csv)[0]}.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                avg_epoch_val_loss))
                valid_loss_min = avg_epoch_val_loss

        # Train, eval and save
        for epoch in range(epochs):
            train()
            eval_and_save(epoch)

    def load_model_from_file(self, load_from_file='./state_dict15c5.pt'):
        self.model.load_state_dict(torch.load(load_from_file))

    def infer_torch(self, torch_tensor: torch.Tensor, load_from_file='./state_dict15c5.pt'):
        if load_from_file:
            self.load_model_from_file(load_from_file=load_from_file)
        inference_input = torch_tensor.to(self.device)
        return self.model(inference_input)

    def infer(self, sequence:pd.DataFrame, load_from_file=None):
        if load_from_file:
            self.load_model_from_file(load_from_file=load_from_file)
        sequence = self.dataset.normalize_data(data_to_normalize_to_train_data=sequence)
        torch_seq = torch.tensor(sequence.values.astype(np.float32))
        torch_seq = torch_seq.to(self.device)
        torch_out = self.model(torch_seq)
        return np.array(torch_out)
