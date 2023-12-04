## PyTorch Lightning 
PyTorch Lightning 是面向专业AI研究人员和机器学习工程师的深度学习框架，该项目旨在不牺牲大规模性能的情况下获得最大的开发灵活性。框架详见：[Pytorch Lighting](https://lightning.ai/docs/pytorch/stable/)

在本文档中，主要包含以下方面的内容：
- PyTorch Lightning的核心概念
- 模板的文件说明
- 使用模板的注意事项

## PyTorch Lightning 的两个核心概念
### 1. [pytorch_lightning.LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)：该部分用于定义模型的训练、验证、测试步骤、optimizer、lr_scheduler。

Pytorch-Lighting的核心设计思想是“自给自足”。在定义自己的模型时，需要继承pytorch_lightning.LightningModule类，并在定义模型的过程中同时实现如何训练、如何测试、优化器定义等内容。
具体的，这些内容通常由以下几个类函数实现：

* def forward : 定义模型的前向传播过程
* def configure_loss ： 定义训练模型的损失函数
* def training_step ： 定义每个batch的训练步骤，在该函数中主要返回训练Loss，用于更新模型参数
* def on_train_epoch_end ： 定义模型在训练时，一个epoch结束时进行的操作
* def validation_step ： 定义每个batch的验证步骤，包括输入batch数据，并得到验证loss等
* def on_validation_epoch_end ：定义模型验证时，一个epoch结束时的操作
* def test_step ： 定义每个batch的测试步骤，基本和validation_step类似
* def on_test_epoch_end ： 定义模型测试时，一个epoch结束时的操作。如计算一个epoch的总体准确率等操作
* def predict_step ： 用于定义模型预测时的操作，通常用于模型推理阶段
* def configure_optimizers ： 定义训练过程中，更新模型所使用的优化器。如SGD,Adam，AdamW等。此外，在该函数中还可以定义用于动态调整学习率的lr_scheduler，如StepLR， MultiStepLR， CosineAnnealingLR等

### 2. [pytorch_lightning.LightningDataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule)： 
该部分定义用于训练、验证和测试的数据集和对应的DataLoader。通常由三个类函数构成：
* def setup : 通常用于初始化Dataset。或者定义模型在训练（fit）和测试（test）阶段所使用的不同数据集。
* def train_dataloader : 利用返回setup函数中定义的训练、验证和测试数据集，定义模型的训练dataloader
* def val_dataloader : 利用返回setup函数中定义的训练、验证和测试数据集，定义模型的验证dataloader
* def test_dataloader : 利用返回setup函数中定义的训练、验证和测试数据集，定义模型的测试dataloader

## 定义的项目模板
本项目提供了一个易用于大型项目、容易迁移、易于复用的模板。 基于该模板，我们需要做的，就是像填空一样，填模板中的这些函数。从而，只需要将精力放在定义模型结构和数据集上，
而无需定义优化器，避免繁杂的中间处理流程。 自定义pytorch lighting模板文件结构如下：
```
root-
	|-data
		|-__init__.py
		|-data_interface.py
		|-xxxstandard_data1.py
		|-xxxstandard_data2.py
		|-...
	|-example_Minist_data
	|-model
		|-__init__.py
		|-model_interface.py
		|-xxxstandard_net1.py
		|-xxxstandard_net2.py
		|-...
	|-utils
        |-__init__.py
        |-xxxutils1.py
        |-xxxutils2.py
        |-...
    |-Logs
	|-main.py
```
- 模板文件说明：其中[data](data)包用于为模型提供训练、验证和测试数据集。[model](model)包用提供自定义模型。[utils](utils)包用于提供模型评估指标、或常用函数等通用模块。[Logs](Logs)用于存储模型训练的日志。
[main.py](main.py)用于模型训练所需的callback函数、实例化数据和模型接口，控制超参数。[example_Minist_data](example_Minist_data)example_Minist_data文件夹提供了MNIST手写数字识别数据集。
同时，在[data](data)包下实现了其对应的训练、验证和测试Dataset，分别为[example_traindata.py](data%2Fexample_traindata.py)、
[example_evaldata.py](data%2Fexample_evaldata.py)和[example_testdata.py](data%2Fexample_testdata.py)。在定义自己的数据集时，可参考它们进行实现。
在model包下提供了自定义模型的基本格式文件[standard_net.py](model%2Fstandard_net.py)和一个用于MNIST手写数字识别的示例模型[example_net.py](model%2Fexample_net.py)。
在定义自己的模型时，可参考它们实现。

- 有关模板更加详细的信息如下：
如果对每个模型直接上pl.LightningModule，对于已有项目、别人的代码等的转换将相当耗时。另外，这样的话，你需要给每个模型都加上一些相似的代码，
如training_step，validation_step。显然，这并不是我们想要的，如果真的这样做，不但不易于维护，反而可能会更加杂乱。
同理，如果把每个数据集类都直接转换成pl的DataModule，也会面临相似的问题。基于这样的考量，我建议使用上述架构：

  - 主目录下只放一个main.py文件。
  - data和model两个文件夹中放入__init__.py文件，做成包。这样方便导入。两个init文件分别是：
    - from .data_interface import DInterface
    - from .model_interface import MInterface
  - utils文件夹中也加入__init__.py文件，做成包。在该包中，可以定义好常用的函数。如评估指标函数。
  - 在data_interface中建立一个class DInterface(pl.LightningDataModule):用作所有数据集文件的接口。__init__()函数中import相应Dataset类，setup()进行实例化，并老老实实加入所需要的的train_dataloader, val_dataloader, test_dataloader函数。 这些函数往往都是相似的，可以用几个输入args控制不同的部分。
  - 同理，在model_interface中建立class MInterface(pl.LightningModule):类，作为模型的中间接口。__init__()函数中import相应模型类，然后老老实实加入configure_optimizers, training_step, validation_step等函数，用一个接口类控制所有模型。不同部分使用输入参数控制。
  - main.py函数只负责：
    - 定义parser，添加parse项
    - 选好需要的callback函数们
    - 实例化MInterface, DInterface, Trainer

## 使用模板的注意事项
- 在使用该模板定义自己实验中需要的数据集时，仅需要参考[standard_data.py](data%2Fstandard_data.py)的样式定义多个数据集，并在[main.py](main.py)的train_dataset、eval_datasets和test_datasets参数中指定对应的数据集。
通常，我们在一个数据集上训练模型，并需要在多个验证集或测试集上评估模型性能。因此，本模板考虑了这一点，在验证集和测试集中，能以列表的形式指定多个验证集和测试集，模型会自动在多个数据集上执行验证，并将结果保存到logs文件中。
- 在model或data包中，一个model或Dataset需在单独的.py文件中定义。同时，模型和Dataset的命名规则必须与模板文件的相同。
即.py文件的命名和定义的模型名称或Dataset的名称对应，如[standard_net.py](model%2Fstandard_net.py)与class <font color=Green>StandardData</font>(data.Dataset)。
因为，所使用的model和Dataset需要通过[main.py](main.py)中的超参数指定，并通过data_interface.py[data_interface.py](data%2Fdata_interface.py)和model_interface.py[model_interface.py](model%2Fmodel_interface.py)中的load_data_module、load_model和instancialize函数实例化。
- 用于控制dataloader的num_workers、batch_size等参数可以直接在[main.py](main.py)中直接添加或修改，并通过main(args)中的args传递到[data_interface.py](data%2Fdata_interface.py)中的DataLoader中。对应的，模型结构的超参数、
优化器、损失函数、动态调整学习率的lr_scheduler的参数也通过args传递到[model_interface.py](model%2Fmodel_interface.py)的pl.LightningModule中。 
这种方式无需再重复修改pl.LightningModule中的其它代码，仅需要我们定义自己的模型和数据集。
- 当[model_interface.py](model%2Fmodel_interface.py)不包含你所需的损失函数、optimizer、lr_scheduler时，你可以在configure_loss、configure_optimizers函数中自行添加。
- main.py中定义了常用的callbacks函数，如
  - EarlyStopping： 在模型训练过程中，用于监测某个指标，当指标不再增加或减少时，停止模型训练。
  当在[main.py](main.py)中指定使用多个验证或测试集时，需要修改该回调函数中对应的监测指标，否则会出现错误。
  - ModelCheckpoint： 在模型训练过程中，用于监测某个指标，保存该指标值达到最大或最小时的模型。当在[main.py](main.py)中指定使用多个验证或测试集时，需要修改该回调函数中对应的监测指标，否则会出现错误。
  - ModelSummary：展示模型的细节信息
  - LearningRateMonitor： 当使用了lr_scheduler时，该回调函数用于监测学习率的变化情况。

## [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/pages/quickstart.html)
TorchMetrics 最初是作为 PyTorch Lightning 的一部分而创建的，PyTorch Lightning 是一个强大的深度学习研究框架，旨在无需模板即可扩展模型。
TorchMetrics 是 100+ PyTorch 指标实现和易于使用的 API 的集合，用于创建自定义指标。
虽然 TorchMetrics 是为与原生 PyTorch 一起使用而构建的，但将 TorchMetrics 与 Lightning 结合使用可提供额外的好处：
- 模块化度量标准在 LightningModule 中正确定义后，会自动放置在正确的设备上。这意味着您的数据将始终与度量值放置在同一设备上。无需再调用!.to(device)
- 原生的支持使用 LightningModule 内的 self.log 在 Lightning 中记录metric。
- metric 的.reset()方法的度量在一个epoch结束后自动被调用

## 实现自己的metrics
如果你想使用一个还不被支持的指标，你可以使用TorchMetrics的API来实现你自己的自定义指标，只需子类化torchmetrics.Metric并实现以下方法:
1. __init__()：每个状态变量都应该使用self.add_state(…)调用。
2. update()：任何需要更新内部度量状态的代码。
3. compute()：从度量值的状态计算一个最终值。
