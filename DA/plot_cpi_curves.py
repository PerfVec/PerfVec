import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from os.path import basename
from os.path import splitext
import sys
import pylab


def extract_data(filename, interval=1, xmax=1, skip=0):
  data = torch.load(filename, map_location=torch.device('cpu'))
  data = data.detach().numpy()
  print(data.shape)
  size = data.shape[0] - skip
  true = data[:, 0, 0]
  pred = data[:, 1, 0]
  #print(true)
  #print(pred)
  assert size % interval == 0 and interval > 0
  jump = xmax / (190.0 / interval)
  xmax_true = jump * size / interval
  x = np.arange(0,xmax_true,jump)
  y1 = true[skip::interval]
  y2 = pred[skip::interval]
  for i in range(1, interval):
    y1 += true[skip+i::interval]
    y2 += pred[skip+i::interval]
  y1 /= 1024 * 512
  y2 /= 1024 * 512
  return x, y1, y2


def plot_cpi_curves(args, for_slides=False):
  if len(args[0]) > 4 and args[0][3] == '.':
    output_name = args[0][0:3] + '_cpis'
  else:
    output_name = args[0] + '_cpis'
  fig_h = 6
  mpl.rcParams['text.usetex'] = True
  font = {'size' : 36}
  plt.rc('font', **font)
  fig, ax = plt.subplots(figsize=(10, fig_h), dpi=100)
  colors = cm.rainbow(np.linspace(0, 1, len(args)))
  color = iter(colors)
  for i in range(1, len(args)):
    file_name = args[i]
    c = next(color)

    x, y1, y2 = extract_data(file_name, 1, 100)
    if i == 1:
      ax.plot(x, y1, c=c, linewidth=2.5)
      c = next(color)
    ax.plot(x, y2, c=c, linewidth=2.5)
    #ax.plot(x, y2 - y1, c=c, linewidth=2.5, linestyle='dotted')

  #ax.scatter(x, y, c='b')
  ax.set_xlabel('Million instructions')
  ax.set_xlim(0, 100)
  ax.set_title(args[0], fontdict={'size': 48})
  ax.set_ylabel('CPI')
  ax.yaxis.grid(True)

  fig.tight_layout()
  fig.savefig('fig/' + output_name + '.pdf')
  #plt.show()
  plt.close()


def plot_legend(args, for_slides=False):
  mpl.rcParams['text.usetex'] = True
  font = {'size' : 36}
  plt.rc('font', **font)
  fig, ax = plt.subplots(figsize=(30, 6), dpi=100)
  #colors = cm.rainbow(np.linspace(0, 1, 2*(len(args) - 1)-1))
  colors = cm.rainbow(np.linspace(0, 1, len(args) - 1))
  #colors = np.concatenate((colors[0::2], colors[1::2]), axis=0)
  color = iter(colors)
  for i in range(1, len(args)):
    file_name = args[i]
    c = next(color)
    if 'true' in file_name:
      label = 'gem5'
    elif 'E1DNet' in file_name:
      label = 'RB7'
    elif 'CNN3' in file_name:
      #str_idx = file_name.find('CNN')
      #label = file_name[str_idx:str_idx+9]
      label = 'C3'
    elif 'CNN5' in file_name:
      label = '5C'
    elif 'CNN7' in file_name:
      label = 'C7'
    elif 'InsLSTM' in file_name:
      label = 'LSTM2'
    else:
      assert 0

    x, y = extract_data(file_name, 1, 100)
    ax.plot(x, y, c=c, linewidth=2.5, label=label)
    if 'true' in file_name:
      true_y = y
    elif not for_slides:
      #c = next(color)
      label += ' error'
      ax.plot(x, y - true_y, c=c, linewidth=2.5, linestyle='dotted', label=label)

  ax.legend(bbox_to_anchor=(0.5, 1.02), loc='upper center', borderaxespad=0, ncol=2*(len(args) - 1)-1)

  figlegend = plt.figure(figsize=(60, 2), dpi=100)
  axlegend = figlegend.add_subplot(111)
  #axlegend.legend(*ax.get_legend_handles_labels(), loc='center')
  legend = axlegend.legend(*ax.get_legend_handles_labels(), loc='center', borderaxespad=0, ncol=2*(len(args) - 1)-1)
  legend.get_frame().set_linewidth(2)
  legend.get_frame().set_edgecolor("black")
  plt.gca().set_axis_off()
  figlegend.tight_layout()
  if for_slides:
    figlegend.savefig('slides/cpis_legend.png')
  else:
    figlegend.savefig('fig/cpis_legend.pdf')
  plt.close()


if __name__ == "__main__":
  if (len(sys.argv) < 3):
    print("Usage: ./plot.py <name> <file>")
    sys.exit(0)
  plot_cpi_curves(sys.argv[1:])
