# 爬虫项目py代码

## 运行环境

  * Python=3.7
  
  > 可使用`conda create -n [环境名] python=3.7`创建新环境避免包污染

  > 使用`source conda activate [环境名]`激活环境并运行下面程序

  > 定时部署可使用虚拟环境的Python可执行文件的绝对路径进行运行
    
  * Conda依赖
    
    > 可尝试使用 `conda install --file requirements_conda.txt` 安装Conda依赖

      - numpy
      
      - requests
            
      - fire
      
      - fiona
  
  * pypi依赖
    
    > 可尝试使用 `pip install -r requirements.txt` 安装pip依赖
    
      - fire

      - requests-html

      - demjson



## 调用方式

### 命令行调用(定时业务化运行)

> 命令行调用方式为按需调用，非常驻进程

  * 执行 `python manage.py run [module]`

## 目录结构
  
  * `radar_spider_decode` 中央气象台网站爬取云图、雷达
  
  * `radar_spider_zgtqw` 中国天气网网站爬取云图、雷达
    