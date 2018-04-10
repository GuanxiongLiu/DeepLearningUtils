import torch
import torchvision
import torchvision.transforms as transforms
from demo_model import Net
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable




if __name__ == '__main__':
	# prepare data loader
	transform = transforms.Compose(
		[transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
	                                        download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
	                                          shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
	                                       download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=4,
	                                         shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat',
	           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	# prepare model and optimizer
	net = Net()
	net.cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	# start training
	for epoch in range(2):
		running_loss = 0
		for i, data in enumerate(trainloader, 0):
			# get inputs
			X, y_ = data
			# wrap to variable
			#X, y_ = Variable(X), Variable(y_)
			X, y_ = Variable(X.cuda()), Variable(y_.cuda())
			# zero the gradients
			optimizer.zero_grad()
			# forward + backward + optimize
			y = net(X)
			loss = criterion(y, y_)
			loss.backward()
			optimizer.step()
			# print statistics
			running_loss += loss.data[0]
			if i % 2000 == 1999:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0
	# end of training
	print('Finished Training')

	# test performance
	correct = 0
	total = 0
	for data in testloader:
		images, labels = data
		outputs = net(Variable(images.cuda()))
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted.cpu().numpy() == labels.cpu().numpy()).sum()

	print('Accuracy of the network on the 10000 test images: %d %%' % (
	    100 * correct / total))