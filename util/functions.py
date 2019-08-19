from segmentation_models_pytorch.utils.functions import *
import numpy as np
import pdb


def confusion_matrix(predicted, target, num_classes, normalized=False):
	"""Computes the confusion matrix
	The shape of the confusion matrix is K x K, where K is the number
	of classes.
	Keyword arguments:
	- predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
	predicted scores obtained from the model for N examples and K classes,
	or an N-tensor/array of integer values between 0 and K-1.
	- target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
	ground-truth classes for N examples and K classes, or an N-tensor/array
	of integer values between 0 and K-1.
	"""
	# If target and/or predicted are tensors, convert them to numpy arrays
	# conf_metric = np.ndarray((num_classes, num_classes), dtype=np.int32)
	if torch.is_tensor(predicted):
		predicted = predicted.cpu().detach().numpy()
	if torch.is_tensor(target):
		target = target.cpu().detach().numpy()

	# pdb.set_trace()
	assert predicted.shape[0] == target.shape[0], \
		'number of targets and predicted outputs do not match'

	if np.ndim(predicted) != 1:
		assert predicted.shape[1] == num_classes, \
			'number of predictions does not match size of confusion matrix'
		predicted = np.argmax(predicted, 1)
	else:
		assert (predicted.max() < num_classes) and (predicted.min() >= 0), \
			'predicted values are not between 0 and k-1'

	if np.ndim(target) != 1:
		assert target.shape[1] == num_classes, \
			'Onehot target does not match size of confusion matrix'
		# pdb.set_trace()
		assert (target >= 0).all() and (target <= 1).all(), \
			'in one-hot encoding, target values should be 0 or 1'
		assert (target.sum(1) == 1).all(), \
			'multi-label setting is not supported'
		target = np.argmax(target, 1)
	else:
		assert (target.max() < num_classes) and (target.min() >= 0), \
			'target values are not between 0 and k-1'

	# hack for bincounting 2 arrays together
	x = predicted + num_classes * target
	bincount_2d = np.bincount(
		x.astype(np.int32), minlength=num_classes ** 2)
	assert bincount_2d.size == num_classes ** 2
	conf = bincount_2d.reshape((num_classes, num_classes))
	if normalized:
		conf = conf.astype(np.float32)
		return conf / conf.sum(1).clip(min=1e-12)[:, None]
	else:
		return conf
