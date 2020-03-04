import matplotlib.pyplot as plt
from executor import *


class Visualizer:
    colors = ['blue', 'black', 'red']

    def __init__(self):
        self.date_pred_targ_dict: dict = dict()

    def add(self, timestamp, pred, targ, color='red'):
        pred_len = pred.shape[1]
        targ_len = targ.shape[1]
        self.date_pred_targ_dict[color] = pd.concat([self.date_pred_targ_dict.get(color, pd.DataFrame()),
                                                     pd.concat([pd.DataFrame(timestamp),
                                                                pd.DataFrame(pred, columns=[f'predicted{i}' for i in
                                                                                            range(pred_len)]),
                                                                pd.DataFrame(targ, columns=[f'target{i}' for i in
                                                                                            range(targ_len)])],
                                                               axis=1)])


_exec = Executor()
_exec.load_model_from_file()
visualizer = Visualizer()
totalIndexes = 0
_exec.model.eval()

# VALIDATE
for val_in, val_out in _exec.dataset.val_loader:
    # Forward pass to get output/logits
    outputs = _exec.infer_torch(torch_tensor=val_in)
    visualizer.add(np.arange(len(val_out)) + totalIndexes, np.array(outputs), np.array(val_out[:, 0].view(-1, 1)),
                   color='blue')
    totalIndexes += len(val_out)

mse_losses = []
mae_losses = []
kldiv_losses = []

# TEST
for test_in, test_out in _exec.dataset.test_loader:
    # Forward pass to get output/logits
    test_in = test_in.to(_exec.device)
    test_out = test_out.to(_exec.device)
    outputs = _exec.infer_torch(torch_tensor=test_in)

    visualizer.add(np.arange(len(test_out)) + totalIndexes, np.array(outputs), np.array(test_out[:, 0].view(-1, 1)))
    totalIndexes += len(test_out)

    mse_losses.append(nn.MSELoss()(outputs, test_out[:, 0].view(-1, 1)).item())
    mae_losses.append(nn.L1Loss()(outputs, test_out[:, 0].view(-1, 1)).item())
    kldiv_losses.append(nn.KLDivLoss()(outputs, test_out[:, 0].view(-1, 1)).item())

print("MSE loss: {:.8f}".format(np.mean(mse_losses)))
print("MAE loss: {:.8f}".format(np.mean(mae_losses)))
print("KLDiv loss: {:.8f}".format(np.mean(kldiv_losses)))


# predictedMax = 'predicted0'
# # predictedMin = 'predicted1'
# targetMax = 'target0'
# # targetMin = 'target1'
# plot_variation_scale=100
#
# fig, ax = plt.subplots(1)
# color_df = visualizer.date_pred_targ_dict.get('blue', pd.DataFrame())
# ax0 = ax
# ax0.scatter(color_df.iloc[:,0], color_df[[predictedMax]]*plot_variation_scale, color='b')
# ax0.plot(color_df.iloc[:,0], color_df[[predictedMax]]*plot_variation_scale, color='b')
# ax0.scatter(color_df.iloc[:,0], color_df[[targetMax]]*plot_variation_scale, color='g')
# ax0.plot(color_df.iloc[:,0], color_df[[targetMax]]*plot_variation_scale, color='g')
# fig.show()
#
# fig, ax = plt.subplots(1)
# color_df = visualizer.date_pred_targ_dict.get('red', pd.DataFrame())
# ax0 = ax
# ax0.scatter(color_df.iloc[:,0], color_df[[predictedMax]]*plot_variation_scale, color='b')
# ax0.plot(color_df.iloc[:,0], color_df[[predictedMax]]*plot_variation_scale, color='b')
# ax0.scatter(color_df.iloc[:,0], color_df[[targetMax]]*plot_variation_scale, color='g')
# ax0.plot(color_df.iloc[:,0], color_df[[targetMax]]*plot_variation_scale, color='g')
# fig.show()
#
# fig, ax = plt.subplots(1)
# ax0 = ax
# # ax1 = ax[1]
# starting_prevday_index = totalIndexes - len(color_df)
# ax0.scatter(color_df.iloc[:,0], color_df[[targetMax]], color='g')
# ax0.plot(color_df.iloc[:,0], color_df[[targetMax]], color='g')
# # plot the difference between the next closing price and the current closing price
# price_variation_df = dataset.test_df.shift(-1)[[CryptoDataset.CLOSING_PRICE]] - dataset.test_df[[CryptoDataset.CLOSING_PRICE]]
# price_variation_df = price_variation_df[SEQUENCE_SIZE-1:]
# ax0.scatter(color_df.iloc[:,0], price_variation_df[[CryptoDataset.CLOSING_PRICE]] * 10000, color='y')
# ax0.plot(color_df.iloc[:,0], price_variation_df[[CryptoDataset.CLOSING_PRICE]] * 10000, color='y')
# # plot the prediction logarithm
# ax0.scatter(color_df.iloc[:,0], color_df[[predictedMax]], color='b')
# ax0.plot(color_df.iloc[:,0], color_df[[predictedMax]], color='b')
# fig.show()