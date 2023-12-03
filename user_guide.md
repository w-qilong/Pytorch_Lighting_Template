## 自定义pytorch lighting模板文件结构
```
root-
	|-data
		|-__init__.py
		|-data_interface.py
		|-xxxstandard_data1.py
		|-xxxstandard_data2.py
		|-...
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
	|-main.py
```

### 在使用该模板时，可按照本文档进行修改
## data包
用于定义训练、验证和测试数据集

## model包
定义模型结构

## main包
控制模型训练