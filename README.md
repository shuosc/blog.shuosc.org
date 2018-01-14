# 上海大学开源社区官方博客

[![pipeline status](https://git.shuosc.org/shuosc/blog.shuosc.org/badges/master/pipeline.svg)](https://git.shuosc.org/shuosc/blog.shuosc.org/commits/master)


[https://blog.shuosc.org](https://blog.shuosc.org)

```bash
$ git push origin master #推送本地更新到github
$ git push gitlab master #推送本地更新到gitlab并更新服务器内容
```
## 注意事项：
    
    * 请为所有的文章加上文章摘要 
        [文章摘要书写方式](https://github.com/bulandent/hexo-theme-bubuzou/blob/master/doc/doc-zh.md)
    * 文章命名规则： 【活动-年份-学期-周数】活动名称
    * tag命名规则，tag需要尽可能加上文章主题相关的关键字，时间（例如17-18秋季学期），主讲人等
    * author请改为自己的名字
    * 请在本次进行`git push`之前先使用`hexo g`生成一下`abbrlink`