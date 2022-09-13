import plot

# choose_accuracy

f = ['checkpoints/8_30_15H/r50_emore_hist.json']

save_path = '/'.join(f[0].split('/')[:-1])
print(save_path)


import json
hh = {}
history = f
for pp in history:
    with open(pp, "r") as ff:
        aa = json.load(ff)
    for kk, vv in aa.items():
        hh.setdefault(kk, []).extend(vv)

print(hh.keys())


plot.choose_accuracy(f,metric_key='lfw')
a,b = plot.hist_plot_split(f,save=save_path + '/result.png')

print(a)
print(b)
#plot.arrays_plot(a, b)

h = [hh['loss'], hh['accuracy'], hh['lfw']]

def plot_hist(h):
    h_l, h_a, h_v_a = h
    import matplotlib.pyplot as plt

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(h_l, 'y', label='train loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(h_a, 'b', label='train acc')
    acc_ax.plot(h_v_a, 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper left')
#    plt.savefig(self.folder_path + '/train.png')
    plt.show()

#plot_hist(h)



