_base_ = ['../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py','../_base_/default_runtime.py']

#模型配置
model = dict(
	head=dict(
		num_classes=5,topk = (1,)
	))

#数据配置
data = dict(samples_per_gpu = 32,workers_per_gpu = 2,
	train = dict(
		data_prefix = '/HOME/scz0be8/run/mmclassification/mmclassification-master/flower/data/flower/train',
		ann_file = '/HOME/scz0be8/run/mmclassification/mmclassification-master/flower/data/flower/train.txt',
		classes = '/HOME/scz0be8/run/mmclassification/mmclassification-master/flower/data/flower/classes.txt'
	),
	val = dict(
		data_prefix = '/HOME/scz0be8/run/mmclassification/mmclassification-master/flower/data/flower/val',
		ann_file = '/HOME/scz0be8/run/mmclassification/mmclassification-master/flower/data/flower/val.txt',
		classes = '/HOME/scz0be8/run/mmclassification/mmclassification-master/flower/data/flower/classes.txt'
	))

#优化器配置
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(policy='step',step=[1])

runner = dict(type='EpochBasedRunner', max_epochs=100)

# 预训练模型
load_from ='/HOME/shenpg/run/mmclassification/mmclassification-master/checkpoints/resnet18_batch256_imagenet_20200708-34ab8f90.pth'
#工作目录
--work_dir='/HOME/scz0be8/run/mmclassification/mmclassification-master/flower/work/resnet'
