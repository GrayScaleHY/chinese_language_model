# chinese_language_model
该工程以transformor为基础搭建拼音转汉字的模型。

功能类似于拼音输入法，如：输入：jin1 tian1 tian1 qi4 ru2 he2  得到：今天天气如何

####################################################################
Requirements
tensorflow-gpu >= 1.14.0
keras >= 2.3.1

##########################################################################
Prepare
data/text_lm.txt # 拼音+汉字 

格式：
yi3 hou4 ni3 shi4 nan2 hai2 zi	以后你是男孩子
lan2 zhou1 na3 you3 mai3 lu4 hu3 qi4 che1 de	兰州哪有买路虎汽车的
kan4 kan4 wo3 de ri4 cheng2 biao3	看看我的日程表
wo3 lao3 po2 shi4 da4 ben4 dan4	我老婆是大笨蛋
wo3 shuo1 ni3 ming2 tian1 zao3 shang4 qi1 dian3 jiao4 wo3 qi3 chuang2	我说你明天早上七点叫我起床
zai4 gei3 wo3 jiang3 ge4 xiao4 hua4 hao3 ma	再给我讲个笑话好吗
bo1 fang4 ge1 qu1 zui4 xuan4 min2 zu2 feng1	播放歌曲最炫民族风
tui4 chu1 dang1 qian2 bo1 fang4 mo2 shi4	退出当前播放模式

################################################################################
Train

python train_lm.py -c config/conf_lm.yaml

注意修改config文件。

#############################################################################
Eval

python test_lm.py -c save_models/11-03-01-49-09-logs/config/config.yaml

在save_models中保存了一个已经训练好的模型。运行text_lm.py时，打印结果为：jin1 tian1 tian1 qi4 ru2 he2  -->  今天天气如何

