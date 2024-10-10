import matplotlib.pyplot as plt
import numpy as np

def draw_batch_cyclic_old():
    # 创建数据
    x = [1,4,8,16,32,64]
    y1 = [17.93,4.64,2.52,1.47,0.97,0.70]  # 第一条线的数据
    y2 = [19.85,6.32,4.34,3.13,2.53,2.15]  # 第二条线的数据
    y3 = [21.81,8.97,6.50,5.28, 4.53,4.15]  # 第三条线的数据
    y4 = [38.94,23.87,20.57,18.89,18.05,17.68]  # 第四条线的数据

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, color = '#ecac27',label='12', marker='o')
    plt.plot(x, y2, color = '#79902d',label='14', marker='s')
    plt.plot(x, y3, color = '#9a4942',label='15', marker='^')
    plt.plot(x, y4, color = '#6da7ca',label='17', marker='D')

    # 添加标题和坐标轴标签
    #plt.title('Four Lines Plot')
    plt.xlabel('batch size')
    plt.ylabel('avarage time')

    # 显示图例
    plt.legend()

    # 显示网格
    #plt.grid(True)

    # 显示图形
    plt.show()
    plt.savefig("four_lines_plot.pdf")

def normalize_array(arr):
    first_value = arr[0]
    return [value / first_value for value in arr]

# 处理每个数组


def draw_batch_cyclic_new():
    x = [1,2,5,8,10,20,30,40,50]
    x12 = [17.99,9.28,3.94,2.55,2.17,1.34,1.02,0.87,0.79]
    x15 = [21.78,13.30,7.97,6.50,6.13,5.00,4.62,4.40,4.28]
    x17 = [39.19,29.16,22.58,20.53,19.97,18.57,18.15,17.82,17.74]
    x18 = [63.11,55.18,47.04,44.85,44.13,42.79,42.19,42.05,41.84]
    x21 = [368.10,353.30,342.49, 339.47,335.31,336.17,335.62,335.30,332.62]
    x23 = [1458.31,1440.21, 1448.66,1445.80,1426.73,1369.39,1363.46,1375.94,1369.0675]

    normalized_x12 = normalize_array(x12)
    normalized_x15 = normalize_array(x15)
    normalized_x17 = normalize_array(x17)
    normalized_x18 = normalize_array(x18)
    normalized_x21 = normalize_array(x21)
    normalized_x23 = normalize_array(x23)

    plt.plot(x, normalized_x12, color = '#9dd0c7',label='12')
    plt.plot(x, normalized_x15, color = '#9180ac',label='14')
    plt.plot(x, normalized_x17, color = '#d9bdd8',label='15')
    plt.plot(x, normalized_x18, color = '#e58579',label='18')
    plt.plot(x, normalized_x21, color = '#8ab1d2',label='21')
    plt.plot(x, normalized_x23, color = '#ecac27',label='23')

    # plt.plot(x, normalized_x12, color = '#9180ac',label='12')
    # plt.plot(x, normalized_x15, color = '#e2745e',label='14')
    # plt.plot(x, normalized_x17, color = '#4a7298',label='15')
    # plt.plot(x, normalized_x18, color = '#9a4942',label='18')
    # plt.plot(x, normalized_x21, color = '#79902d',label='21')
    # plt.plot(x, normalized_x23, color = '#ecac27',label='23')

    plt.xlabel('batch size',fontsize=15)
    plt.ylabel('percentage(%)',fontsize=15)
    # 隐藏所有脊椎
    ax = plt.gca()  # 获取当前坐标轴
    ax.spines[['left','top', 'right','bottom']].set_visible(False)
    ax.grid(axis='x', linestyle='--', linewidth='0.5', color='grey')
    ax.grid(axis='y', linestyle='--', linewidth='0.5', color='grey')
    # 显示图例
    plt.legend(fontsize=15)

# 设置刻度标签的字体大小
    plt.tick_params(axis='both', which='major', labelsize=15)

    # 显示网格
    #plt.grid(True)

    # 显示图形
    plt.show()
    plt.savefig("four_lines_plot.pdf")


def draw_2batch_cyclic_new():
    # 定义原始数组
    x = [1, 2, 5, 8, 10, 20, 30, 40, 50]
    x12 = [17.99, 9.28, 3.94, 2.55, 2.17, 1.34, 1.02, 0.87, 0.79]
    x15 = [21.78, 13.30, 7.97, 6.50, 6.13, 5.00, 4.62, 4.40, 4.28]
    x17 = [39.19, 29.16, 22.58, 20.53, 19.97, 18.57, 18.15, 17.82, 17.74]
    x18 = [63.11, 55.18, 47.04, 44.85, 44.13, 42.79, 42.19, 42.05, 41.84]
    x21 = [368.10, 353.30, 342.49, 339.47, 335.31, 336.17, 335.62, 335.30, 332.62]
    x23 = [1458.31, 1440.21, 1448.66, 1445.80, 1426.73, 1369.39, 1363.46, 1375.94, 1369.0675]

    nx12= [19.97,9.22,3.97,2.56,2.32,1.35,1.05,0.89,0.82]
    nx15=[25.59,13.57,8.11,6.61,6.57,5.09,4.67,4.44,4.29]
    nx18 = [62.56,55.80,46.97,45.05,44.09,42.72,42.28,42.02,41.87]
    nx17=[44.72,29.74, 22.79 ,20.62, 20.07,18.61,18.13,17.89,17.79]
    nx21=[368.83,353.28, 342.67,339.69,338.69,336.52,336.23,333.97,334.42]
    nx23=[1480.17,1462.33,1453.52,1452.30,1446.09,1437.58,1422.47,1401.57,1392.57]
    # 处理每个数组
    normalized_x12 = normalize_array(x12)
    normalized_x15 = normalize_array(x15)
    normalized_x17 = normalize_array(x17)
    normalized_x18 = normalize_array(x18)
    normalized_x21 = normalize_array(x21)
    normalized_x23 = normalize_array(x23)

    normalized_nx12 = normalize_array(nx12)
    normalized_nx15 = normalize_array(nx15)
    normalized_nx17 = normalize_array(nx17)
    normalized_nx18 = normalize_array(nx18)
    normalized_nx21 = normalize_array(nx21)
    normalized_nx23 = normalize_array(nx23)
    # 创建图形
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))  # 创建包含两个子图的图形

    # 绘制第一个子图
    # axs[0].plot(x, normalized_x12, color='#9dd0c7', label='12')
    # axs[0].plot(x, normalized_x15, color='#9180ac', label='14')
    # axs[0].plot(x, normalized_x17, color='#d9bdd8', label='15')
    # axs[0].plot(x, normalized_x18, color='#e58579', label='18')
    # axs[0].plot(x, normalized_x21, color='#8ab1d2', label='21')
    # axs[0].plot(x, normalized_x23, color='#ecac27', label='23')

    axs[0].plot(x, normalized_x12, color = '#aed4e5',label='12')
    axs[0].plot(x, normalized_x15, color = '#81b5d5',label='14')
    axs[0].plot(x, normalized_x17, color = '#5795c7',label='15')
    axs[0].plot(x, normalized_x18, color = '#3371b3',label='18')
    axs[0].plot(x, normalized_x21, color = '#345d82',label='21')
    axs[0].plot(x, normalized_x23, color = '#1e4c9c',label='23')

    # 隐藏所有脊椎
    axs[0].spines[['left', 'top', 'right', 'bottom']].set_visible(False)
    axs[0].grid(axis='x', linestyle='--', linewidth='0.5', color='grey')
    axs[0].grid(axis='y', linestyle='--', linewidth='0.5', color='grey')

    # 添加标题 "a"
    axs[0].text(0.5, -0.2, '(a) cyclic forward NTT', transform=axs[0].transAxes, ha='center', va='center', fontsize=17)

    # 绘制第二个子图
    axs[1].plot(x, normalized_nx12, color='#FFE3A1', label='12')
    axs[1].plot(x, normalized_nx15, color='#FFCD02', label='14')
    axs[1].plot(x, normalized_nx17, color='#F3C846', label='15')
    axs[1].plot(x, normalized_nx18, color='#FEB300', label='18')
    axs[1].plot(x, normalized_nx21, color='#ff9a00', label='21')
    axs[1].plot(x, normalized_nx23, color='#e79201', label='23')

    # 隐藏所有脊椎
    axs[1].spines[['left', 'top', 'right', 'bottom']].set_visible(False)
    axs[1].grid(axis='x', linestyle='--', linewidth='0.5', color='grey')
    axs[1].grid(axis='y', linestyle='--', linewidth='0.5', color='grey')

    # 添加标题 "b"
    axs[1].text(0.5, -0.2, '(b) negative cyclic forward NTT', transform=axs[1].transAxes, ha='center', va='center', fontsize=17)

    # 设置坐标轴标签
    for ax in axs:
        ax.set_xlabel('batch size', fontsize=17)
        ax.set_ylabel('percentage (%)', fontsize=17)
        ax.legend(fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=17)
    # 创建图例并放置在右上方
    # handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, fontsize=10,loc=(0.9, 0.85),ncol=1)

    # 显示图形
    plt.tight_layout()
    plt.show()

    # 保存图形
    plt.savefig("2.pdf", bbox_inches='tight')

def draw_saveup_to_merge_batch():
    x = [1,4,8,16,32,64]
    y1 = [5.420641804,2.526487368,3.326557013,2.80416619,4.149596934,4.364722392]
    y2 = [9.849362688,8.190567854,9.35089266,9.219538572,9.534690811,9.674972655]
    y3 = [8.054972803,8.174286113,8.285879128,8.374346857,9.815429933,10.09716055]
    



    # 假设这是六个月的销售额数据
    months = ['1','4','8','16','32','64']

    # 创建数据点
    x = np.arange(len(months))  # the label locations
    #width = 0.35  # the width of the bars
    plt.figure(figsize=(8, 8))
    # 创建图表
    fig, ax = plt.subplots()
    

    # 绘制折线图
    ax.plot(x, y1, marker='o', linestyle='-', color='black',  markerfacecolor='#4a7298',markersize=13, label='log(n)=15')
    ax.plot(x, y2, marker='s', linestyle='-', color='black', markerfacecolor='#f3c846',markersize=13,label='log(n)=17')
    ax.plot(x, y3, marker='^', linestyle='-', color='black', markerfacecolor='#79902D',markersize=13,label='log(n)=22')


    # 设置x轴的刻度标签为我们的月份
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    

    # 添加标题和坐标轴标签
    ax.set_xlabel('batch size',fontsize=15)
    ax.set_ylabel('save up percentage(%)',fontsize=15)
    
    # 隐藏顶部和右侧的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # 显示图例
    #ax.legend()
    legend = ax.legend(prop={'size': 15}, loc='center right', ncol=1)

    # 显示网格
    #ax.grid(True)
    ax.grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    plt.savefig("mergebatch.pdf",bbox_inches='tight')

    # 展示图表

print("hi")
draw_2batch_cyclic_new()
