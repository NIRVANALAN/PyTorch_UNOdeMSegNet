# unpack given info
import tifffile as tiff
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce


class SlideFetcher:
	def __init__(self, name, labels, where):
		self.name = name
		self.labels = labels
		self.label2where = dict()
		for l, w in zip(labels, where):
			self.label2where[l] = w

	def __getitem__(self, key):
		return self.get_img(key)

	def get_img(self, label):
		return tiff.imread(self.label_path(label))

	def label_path(self, label):
		return self.label2where[label]


# discover slide files under MROOT
'''
slides = dict()
for d in os.listdir(MROOT):
	subpath = os.path.join(MROOT, d)
	for dslide in os.listdir(subpath):
		subpathslide = os.path.join(subpath, dslide)
		labels, where = [], []
		for dimg in os.listdir(subpathslide):
			if dslide in dimg:
				label_name = dimg[len(dslide) + 1:]
				if label_name == 'tif':
					label_name = '__raw__'
				elif label_name.endswith('.tiff'):
					label_name = label_name[:-5]
				labels.append(label_name)
				where.append(os.path.join(subpathslide, dimg))
		slides[dslide] = SlideFetcher(dslide, labels, where)
slides

'''


def visualize(slides):
	allskey = list(slides.keys())[5:8]  # sample only 3 of them
	nlabel = 1
	label2val = {'Background': 0}
	val2label = {0: 'Background'}
	allalllabel = []
	allrimg = []
	for skey in allskey:
		tlabel = slides[skey].labels[3]
		rimg = slides[skey]['__raw__']

		alllabel = np.zeros(rimg.shape, dtype=int)
		for l in slides[skey].labels:
			if l == '__raw__': continue
			limg = slides[skey][l]
			if l not in label2val:
				label2val[l] = nlabel
				val2label[nlabel] = l
				nlabel += 1
			alllabel[limg != 0] = label2val[l]
			del limg

		allalllabel.append(alllabel)
		allrimg.append(rimg)

		N, SZ = len(allskey), 8
		fig, axes = plt.subplots(nrows=N, ncols=2, figsize=(2 * SZ, N * SZ))
		for axs, alllabel, rimg, name in zip(axes, allalllabel, allrimg, allskey):
			axs[1].imshow(alllabel[::8, ::8])
			axs[1].set_title(f'{name} label')
			axs[0].imshow(rimg[::8, ::8], cmap='gray')
			axs[0].set_title(f'{name} raw {rimg.shape}')
		for ax in axes.flatten():
			ax.axis('off')


def alllabel2onehot(y, nclass):
	py = np.zeros(y.shape + (nclass,))
	for c in range(nclass):
		py[..., c][y == c] = 1.0
	py = np.transpose(py, (2, 0, 1))
	return py


# IDX = 1
# ximg = allrimg[IDX]
# ylabel = allalllabel[IDX]
# xname = allskey[IDX]
# print(ximg.shape, ylabel.shape, xname)

def extract_window(ximg, ylabel, lx, rx, ly, ry, nclass, dsr=1):
	if dsr == 1:
		return ximg[lx:rx:dsr, ly:ry:dsr], \
			   ylabel[lx:rx:dsr, ly:ry:dsr], \
			   alllabel2onehot(ylabel[lx:rx, ly:ry], nclass)
	return ximg[lx:rx:dsr, ly:ry:dsr], \
		   ylabel[lx:rx:dsr, ly:ry:dsr], \
		   block_reduce(alllabel2onehot(ylabel[lx:rx, ly:ry], nclass), (1, dsr, dsr), np.mean)


def show_triplet(ximg_sel, ylabel_sel, yprob_sel, C=1, DSR=1):
	SZ = 6
	fig, axs = plt.subplots(ncols=3, figsize=(2 * SZ, N * SZ))
	axs[0].imshow(ximg_sel[::DSR, ::DSR], cmap='gray')
	axs[0].set_title(f'{xname} raw {ximg_sel.shape}')
	axs[1].imshow(ylabel_sel[::DSR, ::DSR])
	axs[1].set_title(f'label')
	axs[2].imshow(yprob_sel[C, ::DSR, ::DSR])
	axs[2].set_title(f'class {C}')
	for ax in axs.flatten(): ax.axis('off')


ximg_dsr, ylabel_dsr, yprob_dsr = extract_window(1600, 3200, 1600, 3200, 8, 16)
print(ximg_dsr.shape, ylabel_dsr.shape, yprob_dsr.shape)


# show_triplet(ximg_dsr, ylabel_dsr, yprob_dsr)


def show_pred(logpx, cap=None):
	M, SZ = 8, 2
	N = (logpx.shape[0] - 1) // M + 1
	fig, axes = plt.subplots(nrows=N, ncols=M, figsize=(SZ * M, SZ * N));
	axes = axes.flatten()
	for ax, lpx in zip(axes, logpx):
		if cap is None:
			ax.imshow(np.exp(lpx))
		else:
			z = np.zeros(lpx.shape);
			z[np.exp(lpx) > cap] = 1.0
			ax.imshow(z)
		ax.axis('off')


ximg_dsr_t = torch.Tensor(ximg_dsr)[None, None, ...]
yprob_dsr_t = torch.Tensor(yprob_dsr)[None, ...]

with torch.no_grad():
	logpx = segnet(ximg_dsr_t).detach().numpy()[0]
show_pred(logpx)
