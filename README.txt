这里面包括“用电器识别“项目。

数据文件（夹）：
    Data_origin文件夹 : 原始数据，里面是5个家庭的总表和分表的数据。
        URL：http://jack-kelly.com/data/
        引用信息：Jack Kelly and William Knottenbelt. The UK-DALE dataset, domestic appliance-level electricity demand and whole-house demand from five UK homes. Scientific Data 2, Article number:150007, 2015, DOI:10.1038/sdata.2015.7
         说明：下载的只是该网页中的“1 second and 6 second data”，里面主要是6s的，只有总表有1s的数据。该网页还可以下载16khz等高频数据。

Python程序：
（每个文件对应一个相同名字的*.spydata文件，里面存的都是最后的结果，可以直接读进来对着相关变量的注释看）
    trial1.py ：　（可忽略）对house5的数据所做的各种处理和实验，包括读入数据、数据处理、标注真实用电器开关、模拟识别。所有数据都读入内存后运算。但是因为trial2做了很多改进，所以可以忽略这个版本。
    trial1_fun.py : （可忽略）trial1.py需要调用的各种自写的functions。

    trial2.py ： 对house1的数据所做的各种处理和实验，包括读入数据、数据处理、标注真实用电器开关、模拟识别。因为数据量太大无法读入内存，中间结果都缓存到了data文件夹里面。虽然IO因此慢了一点，但是因为算法改进了很多，所以整体速度要比trial1快不少。有详细注释。
    trial2_fun.py : trial2.py需要调用的各种自写的functions。

其他文件（夹）：
    
