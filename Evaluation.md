数据简介 About the Data
本次竞赛数据分为两部分：训练集与测试集，其中训练集在竞赛开始时开放给参赛队；测试集在竞赛结束前1天开放给参赛队。

数据说明 Data description
数据形式：金融网络文本（标题和内容）和文本中出现的实体列表。
数据规模：训练集数据量1万条，测试集数据量1万条。
数据类型：中文自然语言文本。
Form: Financial network text (title and content) and entities that appear in the text.
Volume: The training set has 10,000 data and the test set has 10,000 data.
Type: Chinese Natural Language Text.
训练数据全部以csv格式给出。每条数据包括标识号（id）、文本标题（title）、文本内容（text）、实体列表（entity）、是否负面（negative）、 负面主体（key_entity）。不同字段之间以英文逗号分隔，同一字段内实体用英文分号分隔。
All training data are given in CSV format. Each data includes ID, title, text, entity, negative and key entity. Different fields are separated by English commas, and entities in the same field are separated by English semicolons.

字段信息 Field Info	类型 Type	描述 Description
id	String	数据ID Data ID
title	String	文本标题 Text title
text	String	文本内容 Text content
entity	String	给定的实体列表 Given entities
negative	String	文本是否包含负面信息 Does the text contain negative information
key_entity	String	负面主体 Key entity
测试数据也是csv格式，每条数据包括标识号（id）、文本标题（title）、文本内容（text）和实体列表（entity）。不同字段之间以英文逗号分隔，同一字段内实体用英文分号分隔。
The test data is also in CSV format, with each data including id, title, text and entity. Different fields are separated by English commas, and entities in the same field are separated by English semicolons.

字段信息 Field Info	类型 Type	描述 Description
id	String	数据ID Data ID
title	String	文本标题 Text title
text	String	文本内容 Text content
entity	String	给定的实体列表 Given entities
对数据集中字段需要进行简要说明：“title”和“text”至少有一个不为空，部分微博来源数据的“title”和“text”相同。“negative”取值为0或1,1表示是负面，0表示非负面。
Note: Either “title” or “text” fields must be filled. Some of the Weibo sourced data has the same “title” and “text”. The value of “negative” is 0 or 1. 1 stands for negative, and 0 for non-negative.

提交要求 Submission requirements
（1）结果需提交csv格式文件，包括文本标识号（id）、是否负面（negative）、负面主体（key_entity）。不同字段之间以英文逗号分隔，同一字段内实体用英文分号分隔。
The results should be submitted in CSV format, including text id (id), negative and key entity. Different fields are separated by English commas, and entities in the same field are separated by English semicolons.

字段信息 Field Info	类型 Type	描述 Description
id	String	数据ID Data ID
negative	String	文本是否包含负面信息 Does the text contain negative information
key_entity	String	负面主体 Key entity
（2）提交文件中的文本标识号需与下发文件中的标识号一一对应；
The text id of the submitted document should fully match the id in the given document.
（3）评测排名靠前的团队需要提交技术报告。
The top ranked teams need to submit technical reports.

提交示例 Submission sample
id,negative,key_entity
2,1,草根投资;石头理财
…

评测标准 Evaluation criteria
本任务使用
F_1值进行评价，采用负面判定指标F_{s}^{1}、主体判定指标F_{1}^{e}和任务整体得分F_{1}
​
 三个评价指标和相应的三个排行榜进行评价，具体如下：
This task is evaluated using F_1F 1 value. Three evaluation indexes and three rankings are used to evaluate the task. They are negative judgment index F_1 ^ sF1s, entity judgment index F_1 ^ eF1e and overall task score F_1F1.

设n为金融文本总数,
Let n be the total number of financial texts.

T P_{s}表示负面判定正确的文本数量 the number of text correctly recognized as negative

F P_{s}表示非负面文本被误判为负面文本的数量 the number of non-negative text wrongly recognized as negative

F N_{s}表示负面文本错误判断为非负面文本的数量 the number of negative text wrongly recognized as non-negative

\begin{array}{c}
P_{s}=\frac{T P_{s}}{T P_{s}+F P_{s}} \\
R_{s}=\frac{T P_{s}}{T P_{s}+F N_{s}} \\
F_{s}^{1}=\frac{2 P_{s} R_{s}}{P_{s}+R_{s}}
\end{array}


T P_{e i}表示第i条文本中正确识别为主体的数量 the number of entity correctly recognized in No.i text.

F N_{e i}表示第i条文本中未识别出的主体的数量 the number of entity unidentified in No.i text

F P_{e i}表示第i条文本中错误识别为主体的数量 the number of wrongly recognized entities in No.i text.
\begin{array}{c}
P_{e}=\frac{\sum_{i=1}^{n} T P_{e i}}{\sum_{i=1}^{n} T P_{e i}+\sum_{i=1}^{n} F P_{e i}} \\
R_{e}=\frac{\sum_{i=1}^{n} T P_{e i}}{\sum_{i=1}^{n} T P_{e i}+\sum_{i=1}^{n} F N_{e i}} \\
F_{1}^{e}=\frac{2 P_{e} R_{e}}{P_{e}+R_{e}} \\
F_{1}=0.4 * F_{1}^{s}+0.6 * F_{1}^{e}
\end{array}
