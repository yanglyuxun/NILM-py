这里面包括“用电器识别“项目。

数据文件（夹）：
    Data_origin文件夹 : 原始数据，里面是5个家庭的总表和分表的数据。（太大了，我把uk-power下的文件删掉了。所以需要先解压/Data_origin/uk-power-data-zip-download.zip后才能运行后面的python程序）
        URL：http://jack-kelly.com/data/
        引用信息：Jack Kelly and William Knottenbelt. The UK-DALE dataset, domestic appliance-level electricity demand and whole-house demand from five UK homes. Scientific Data 2, Article number:150007, 2015, DOI:10.1038/sdata.2015.7
         说明：下载的只是该网页中的“1 second and 6 second data”，里面主要是6s的，只有总表有1s的数据。该网页还可以下载16khz等高频数据。

Python程序：
（每个文件对应一个相同名字的*.spydata文件，里面存的都是最后的结果，可以直接读进来对着相关变量的注释看）
    trial1.py ：（可忽略）对house5的数据所做的各种处理和实验，包括读入数据、数据处理、标注真实用电器开关、模拟识别。所有数据都读入内存后运算。但是因为trial2做了很多改进，所以可以忽略这个版本。
    trial1_fun.py : （可忽略）trial1.py需要调用的各种自写的functions。

    trial2.py ： 对house1的数据所做的各种处理和实验，包括读入数据、数据处理、标注真实用电器开关、模拟识别。因为数据量太大无法读入内存，中间结果都缓存到了data文件夹里面，随用随读。虽然IO因此慢了一点，但是因为算法改进了很多，所以整体速度要比trial1快不少。有详细注释。
    trial2_fun.py : trial2.py需要调用的各种自写的functions。
    trial2_para_tuning.py：XGBooost调参方法，主要是通过多次的CVsearch来调参。内有参考的资料和修改后的代码。本身不会被调用，但是结果放在trial2.py中使用。有详细注释。

    trial3.py ： 对house1中的1s1个的数据做处理。但是没有做完，在读入数据阶段遇到各种问题，所以仅供参考。有详细注释。

其他文件（夹）：
    data文件夹：这是trial2.py中生成的缓存pickle文件（太大了，已经全部删除。只要运行trial2.py就可以重建）
    img文件夹：这是trial2.py中作图生成的文件
    new_house5文件夹：这个文件夹已经废弃
