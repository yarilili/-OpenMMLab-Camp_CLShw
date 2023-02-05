_base_ = ['./mobilenet-v3-small_8xb128_in1k.py']

#模型配置
model = dict(
    backbone=dict(type='MobileNetV3'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
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


dataset_type = 'CustomDataset'
data_preprocessor = dict(num_classes=5)
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-small_8xb128_in1k_20221114-bd1bfcde.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=5, topk=(1,)))

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root='data',
        data_prefix='train',
        pipeline={{_base_.train_pipeline}}
    ),
)

val_dataloader = dict(
    batch_size = 32,
    num_workers = 2,
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root='data',
        data_prefix='val',
        pipeline={{_base_.test_pipeline}}
    ),

)

#预训练模型
load_from = '/HOME/scz0be8/run/mmclassification/mmclassification-master/checkpoints/mobilenet_v3_small-8427ecf0.pth'
resume_from = None
workflow = [('train', 1)]

#优化器
optimizer = dict(type='SGD', lr=0.0001, momentum=0.2, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    step=[5,15])

runner = dict(type='EpochBasedRunner', max_epochs=20)
#工作目录
--work_dir='/HOME/scz0be8/run/mmclassification/mmclassification-master/flower/work/mobilenet'
