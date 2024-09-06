## 文件目录结构
benchmark下对应着gpu实现的main函数

example对应着README文件夹下Testing GPU NTTs vs CPU NTTs的内容
example/cpu_ntt对应README下的Testing CPU Merge & 4-Step NTT vs Schoolbook Polynomial Multiplication

## 修改思路
先确认下CPU实现下的正确性(已完成)


## GPU端的修改思路
### ntt
直接删掉最后一步的GPU_Transpose即可
