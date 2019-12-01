# BertTask
使用bert预训练好的模型来fine-tuning, 实现自然语言处理的一些下游任务，包括：<br>
* 分类任务
* 对话系统
# fine-tune一般过程
### 1.处理原始数据
在做文本分类任务时，需要修改`run_classifier.py`文件，在其中添加自己的数据处理类；在此建议修改`run_classifier.py`文件前先对自己的数据进行处理，将数据格式修改成`.tsv`形式。tsv格式即一个样本中标签和文本间用`tab`隔开，对于对话系统类，两个句子间也用`tab`隔开。如下图所示<br>
![](https://github.com/orangerfun/BertTask/raw/master/tsv.png)

在`run_classifier.py`文件中有一个`DataProcessor`类，该类中有一个`_read_tsv`方法，若事先未将数据修改成上述的tsv文件格式，需要修改该方法，若已经将数据处理成tsv数据格式，则该方法不需要修改<br>
![](https://github.com/orangerfun/BertTask/raw/master/readtsv.png)
`_read_tsv`方法返回的是一个双重列表，`[[1],[2],[3]]` , `[1]` 是一个列表，元素是标签，句子（都是字符串）<br>
### 2.定义自己的数据处理类
接下来在`run_classifier.py`文件中定义自己的数据处理类，暂且将类名命为`MyProcessor`主要需要修改的地方如图中标注所示（【】标注的地方）<br>
![](https://github.com/orangerfun/BertTask/raw/master/myproce.png)

* `[1]`如果处理的文本是中文，需要设置380行，如果是英语，则不需要更改__init__方法<br>
* `[2]`标签需要更改，标签可以是文本形式，可以直接返回标签列表，也可以写一个函数从文本中获取标签，注意返回的是一个列表<br>
* `[3]` `text_a`, `text_b`需要更改（具体根据自己语料更改），如果是分类任务，`text_b=None`, 若是问答系统，text_b是另一句话; 在测试集时（409行）label设置成任意值<br>

**注意：** 如果输入句子不是`tsv`格式，需要自定义函数从样本中获取句子和`label`,此时需将`Myprocessor`类中调用的`_read_tsv`方法（如385行）换成自己定义的方法<br>

从上面我们自定义的数据处理类中可以看出，训练集和验证集是保存在不同文件中的，因此我们需要将我们之前预处理好的数据提前分割成训练集和验证集，并存放在同一个文件夹下面，文件的名称要和类中方法里的名称相同

### 3.将自定义类加入main函数中的字典里
![](https://github.com/orangerfun/BertTask/raw/master/main.png)

### 4.写脚本运行run_classifier.py
![](https://github.com/orangerfun/BertTask/raw/master/script.png)
task_name就是我们定义的数据处理类的键，BERT模型较大，加载时需要较大的内存，如果出现内存溢出的问题，可以适当的降低batch_size的值

### 5.预训练模型
[模型下载地址](https://github.com/googleresearch/bert/blob/master/multilingual.md)<br>
[参数较小的中文预训练模型](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)







