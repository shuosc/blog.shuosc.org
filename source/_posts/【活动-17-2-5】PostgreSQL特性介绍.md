---
title: PostgreSQL特性介绍
author: shuosc
tags: 活动记录
categories: 社区活动
abbrlink: 
date: 2018-01-09 00:00:00
---
本次是开源社区的第五次活动，本次活动很荣幸地请到了 森亿智能 的 CTO @佳能来分享关于PostgreSQL的介绍。

![殷嘉珩学长](https://mmbiz.qpic.cn/mmbiz_jpg/ErNIAficWks18W2zWiceDuWGibtVvQPtUa4bJgQgkTkVUnqIwNVaVTt7lvzNJj716KT4DqQiaYh8RlMOPdG3YHhyBQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

 本次分享内容

1. PG的背景介绍

2. PG vs mysql比较

3. jsonb 文档数据类型（vs mongodb）

4. 多样的索引算法（GIN，GIST，KNN，HASH，BRIN）

5. 自定义函数UDF（python，js，C等）

6. 神器foreign data wrap

7. MPP分布式大数据仓库greenplum

8. PG生态

![](https://mmbiz.qpic.cn/mmbiz_jpg/ErNIAficWks18W2zWiceDuWGibtVvQPtUa4ZvhZmp3icAB7fzDJicHhETnxicgOCWusAX0ickP2GGAII8YGZBiagX5S6pQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![](https://mmbiz.qpic.cn/mmbiz_jpg/ErNIAficWks18W2zWiceDuWGibtVvQPtUa49AHNibXjE4rylIEtbaJanW2ibicJsWDB2e6rTrP474ic71BdJMZQiaPhV1g/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![](http://mmbiz.qpic.cn/mmbiz_jpg/ErNIAficWks18W2zWiceDuWGibtVvQPtUa4yhCf2c8T6XdFZKrPMobw1uSW8gD9Glm33FZOeOlrwdSvHPqElvCjDw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



学长可以说讲解的非常细致，在自己经验上分享了很多，包括之前踩过的坑，以及项目中用到的不同的功能，还有之前做过的各种测试等，大家也听的很仔细了。可以说是收获满满。


最后的问答环节，@cosformula同学询问了学长对MongoDb的看法。学长的回答是，在处理类似于实时有冲突信息插入数据时，会出现插入数据时间顺序出错问题（摘取一部分，详细可以去B站看视频哦~）

  

最后的最后，感谢给大家发冰糖桔和徽章的@cosformula

![](http://mmbiz.qpic.cn/mmbiz_jpg/ErNIAficWks18W2zWiceDuWGibtVvQPtUa4SqKIYp3Abcic2OMiaTqXxOVeJ3hDYWM6KhxjAfSSoXK4EJusK39gtGWw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
![](http://mmbiz.qpic.cn/mmbiz_jpg/ErNIAficWks18W2zWiceDuWGibtVvQPtUa4b2hmKld9ibeTmsTavXwRNhVU9iciaVpcZp5WvegMQncdtGBaa6iaoCicU1g/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)


ps:下次活动的时间地点照旧，欢迎大家来参加~~

| 附录 |
| :------- |
|[本次活动的PPT](https://github.com/shuopensourcecommunity/meta-OSC/raw/master/activities/2017/winter/week-5/postgresql-shu.pptx)|

