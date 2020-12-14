# GUI
## PaintBoard
设置*画板*大小为`[480, 460]`，*背景色*为`white`，默认无内容，不能使用橡皮擦，且初始*鼠标位置*`(0, 0)`。
*画笔*粗细为`10px`颜色`black`，并获取可用的颜色列表。

* `清除` 清空画布，重设背景色
* `画笔` 画笔颜色可改变，粗细也可以改变

## MyWidget
略

`Attention`  `file`main-->`class`MyWidget-->`method`btn_recog_clicked`line187`

    predict = method_index # method_index使用自己相应index的method
    
# Data
`数据集`   通过`save`或者`index`存画的图；

`savepath`是文件夹时存在文件夹里，自动命名，文件夹名建议取相应数字名；`savepath`是文件时直接存成文件。

`识别`    所用图是`recog.jpg`。

# ShortCut
`Ctrl+Q`    退出

`Ctrl+R`    识别

`Ctrl+E`    清除log信息

`Ctrl+S`    通过`index`保存图片

`Ctrl+X`    清除画布

`Ctrl+A`    通过`save`保存图片

# Packages
PyQt5
