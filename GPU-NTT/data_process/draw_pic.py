import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt

def draw_graph(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,line1_label,line2_label,line3_label,line4_label,line5_label,line6_label,line7_label,line8_label,name):

    plt.clf()
    # sorted_pairs_1 = sorted(zip(x1, y1), key=lambda pair: pair[0])
    # x1_sorted, y1_sorted = zip(*sorted_pairs_1)

    # # 对x2, y2数据进行排序
    # sorted_pairs_2 = sorted(zip(x2, y2), key=lambda pair: pair[0])
    # x2_sorted, y2_sorted = zip(*sorted_pairs_2)

    # sorted_pairs_3 = sorted(zip(x3, y3), key=lambda pair: pair[0])
    # x3_sorted, y3_sorted = zip(*sorted_pairs_3)

    # sorted_pairs_4 = sorted(zip(x4, y4), key=lambda pair: pair[0])
    # x4_sorted, y4_sorted = zip(*sorted_pairs_4)

    # sorted_pairs_5 = sorted(zip(x5, y5), key=lambda pair: pair[0])
    # x5_sorted, y5_sorted = zip(*sorted_pairs_5)

    # sorted_pairs_6 = sorted(zip(x6, y6), key=lambda pair: pair[0])
    # x6_sorted, y6_sorted = zip(*sorted_pairs_6)

    # sorted_pairs_7 = sorted(zip(x7, y7), key=lambda pair: pair[0])
    # x7_sorted, y7_sorted = zip(*sorted_pairs_7)


    # 绘制第一条直线
    plt.plot(x1, y1, label=line1_label, color='blue', linestyle='-', marker='o',linewidth=1)

    # 绘制第二条直线
    plt.plot(x2, y2, label=line2_label, color='red', linestyle='--', marker='s',linewidth=1)

    plt.plot(x3, y3, label=line3_label, color='yellow', linestyle='--', marker='s',linewidth=1)

    plt.plot(x4, y4, label=line4_label, color='green', linestyle='--', marker='s',linewidth=1)

    plt.plot(x5, y5, label=line5_label, color='green', linestyle='--', marker='s',linewidth=1)

    plt.plot(x6, y6, label=line6_label, color='green', linestyle='--', marker='s',linewidth=1)

    plt.plot(x7, y7, label=line7_label, color='green', linestyle='--', marker='s',linewidth=1)

    plt.plot(x8, y8, label=line8_label, color='green', linestyle='--', marker='s',linewidth=1)


    


    # 设置图表的标题和坐标轴标签
    plt.title(name)
    plt.xlabel('packet loss(%)')
    plt.ylabel('handshake time(ms)')

    # 显示图例
    plt.legend()

    # 显示网格（可选）
    plt.grid(True)
    plt.savefig(f'./a.png', format='png')

    # 显示图表
    plt.show()


def draw_bar():

    # 示例数据
    categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
    values = [20, 35, 25, 30]

    # 创建一个新的图像
    plt.figure()

    # 绘制水平条形图
    plt.barh(categories, values, color='skyblue')

    # 添加标题和轴标签
    plt.xlabel('Value')
    plt.ylabel('Category')
    plt.title('Horizontal Bar Chart')
    plt.savefig(f'./a.png', format='png')
    # 显示图表
    plt.show()

def draw_2bar():
    # 每个图中的组数
    groups_per_plot = 5
    # 每组中的柱子数
    bars_per_group = 2
    # 柱子的宽度
    bar_width = 0.2
    # 柱子组之间的间隔
    group_gap = 5
    # 图的数量
    num_plots = 4

    # 创建一个从0开始的横坐标位置
    positions = np.arange(groups_per_plot) * (bars_per_group * bar_width + group_gap)

    # 创建随机数据，这里假设每个图的数据都是随机的
    data = np.random.rand(num_plots, groups_per_plot, bars_per_group)

    # 创建颜色列表
    colors = ['skyblue', 'salmon']  # 两个颜色交替

    # 创建画布和子图
    fig, axs = plt.subplots(1, num_plots, figsize=(10, 2))

    # 如果只有一个子图，将其转换为数组以便统一处理
    if num_plots == 1:
        axs = [axs]

    # 绘制横向柱状图
    for ax, plot_data in zip(axs, data):
        for i in range(groups_per_plot):
            # 每组的第一个柱子的位置
            pos = positions[i]
            # 在同一位置绘制两个柱子，通过颜色区分
            for j in range(bars_per_group):
                ax.barh(pos + j * bar_width, plot_data[i, j], bar_width, color=colors[j % len(colors)])
        
        # 只有最左侧的子图保留 y 轴标签
        if ax != axs[0]:
            ax.set_yticklabels([])

    # 设置y轴标签
    axs[0].set_yticks(positions + bar_width / 2)
    axs[0].set_yticklabels(('Group 1', 'Group 2', 'Group 3', 'Group 4'))

    # 显示网格
    for ax in axs:
        ax.grid(False)

    # 调整子图间距
    plt.tight_layout()

    # 展示图形
    plt.show()
    plt.savefig(f'./b.png', format='png')

def draw_22bar():
    # 数据
    data = [
        [[24.5,10.27], [24.4,5.28], [24.475,2.76], [24.5,1.60375],[24.5625,1.03125]],  # 第一个子图的数据
        [[52.6,10.79], [53,5.615], [53,3.3125], [52.875,2.1875],[53.75,1.745625]],  # 第二个子图的数据
        [[113,11.94], [113,6.865], [112.75,4.7775],[114.75,3.65625],[116.75,3.06]],
        [[512,21.7], [516,16.785], [525.5,13.68], [531.375,11.74],[533.5,10.61]],
        # ... 其他子图的数据
    ]

    # 每个图中的组数
    groups_per_plot = len(data[0])
    # 每组中的柱子数
    bars_per_group = len(data[0][0])
    # 柱子的宽度
    bar_width = 0.1
    # 柱子组之间的间隔
    group_gap = 0.05
    # 图的数量
    num_plots = len(data)

    # 创建一个从0开始的横坐标位置
    positions = np.arange(groups_per_plot) * (bars_per_group * bar_width + group_gap)

    # 创建颜色列表
    colors = ['#4a7298', '#f3c846']  # 两个颜色交替

    # 创建画布和子图
    fig, axs = plt.subplots(1, num_plots, figsize=(15, 3))

    # 如果只有一个子图，将其转换为数组以便统一处理
    if num_plots == 1:
        axs = [axs]

    # 绘制横向柱状图
    for ax, plot_data in zip(axs, data):
        for i in range(groups_per_plot):
            # 每组的第一个柱子的位置
            pos = positions[i]
            # 在同一位置绘制两个柱子，通过颜色区分
            for j in range(bars_per_group):
                ax.barh(pos + j * bar_width, plot_data[i][j], bar_width, color=colors[j % len(colors)])
        
        # 只有最左侧的子图保留 y 轴标签
        if ax != axs[0]:
            ax.set_yticklabels([])

    # 设置y轴标签
    axs[0].set_yticks(positions + bar_width / 2)
    axs[0].set_yticklabels(['tower = 1','tower = 2', 'tower = 4', 'tower = 8', 'tower = 16'])

    # 显示网格
    # for ax in axs:
    #     ax.grid(True)

    # 调整子图间距
    plt.tight_layout()

    # 展示图形
    plt.show()
    plt.savefig(f'./a.png', format='png')


def draw_23bar():
    # 数据
    data = [
        [[24.5, 10.27], [24.4, 5.28], [24.475, 2.76], [24.5, 1.60375], [24.5625, 1.03125]],  # 第一个子图的数据
        [[52.6, 10.79], [53, 5.615], [53, 3.3125], [52.875, 2.1875], [53.75, 1.745625]],  # 第二个子图的数据
        [[113, 11.94], [113, 6.865], [112.75, 4.7775], [114.75, 3.65625], [116.75, 3.06]],
        [[512, 21.7], [516, 16.785], [525.5, 13.68], [531.375, 11.74], [533.5, 10.61]],
        # ... 其他子图的数据
    ]

    # 每个图中的组数
    groups_per_plot = len(data[0])
    # 每组中的柱子数
    bars_per_group = len(data[0][0])
    # 柱子的宽度
    bar_width = 0.8
    # 柱子组之间的间隔
    group_gap = 0.04
    # 图的数量
    num_plots = len(data)

    # 创建一个从0开始的横坐标位置
    positions = np.arange(groups_per_plot) * (bars_per_group * bar_width + group_gap)

    # 创建颜色列表
    colors = ['#4a7298', '#f3c846']  # 两个颜色交替

    # 创建画布和子图
    fig, axs = plt.subplots(1, num_plots, figsize=(10, 2))

    # 如果只有一个子图，将其转换为数组以便统一处理
    if num_plots == 1:
        axs = [axs]

    # 定义子图的标题和x轴标签
    titles = ['log(n)=12', 'log(n)=13', 'log(n)=14', 'log(n)=16']  # 这里定义每个子图的标题
    #x_labels = ['x1', 'x2', 'x3', 'x4', 'x5']  # x轴的标签

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=colors[0], label='OpenFHE'),
        plt.Rectangle((0, 0), 1, 1, fc=colors[1], label='cu4stepNTT')
    ]

    # 绘制横向柱状图
    for ax, plot_data, title in zip(axs, data, titles):
        for i in range(groups_per_plot):
            # 每组的第一个柱子的位置
            pos = positions[i]
            # 在同一位置绘制两个柱子，通过颜色区分
            for j in range(bars_per_group):
                ax.barh(pos + j * bar_width, plot_data[i][j], bar_width, color=colors[j % len(colors)])

        # 设置子图的标题
        ax.set_title(title)
        # 设置x轴标签
        #ax.set_xticklabels(x_labels)

        # 只有最左侧的子图保留 y 轴标签
        if ax != axs[0]:
            ax.set_yticklabels([])

    # 设置y轴标签
    # 在最右侧的子图中添加图例
    if ax == axs[-1]:  # 确保是在最右侧的子图中添加图例
        ax.legend(handles=legend_handles, loc='upper right')
    axs[0].set_yticks(positions + bar_width / 2)
    axs[0].set_yticklabels(['tower = 1', 'tower = 2', 'tower = 4', 'tower = 8', 'tower = 16'])
    fig.text(0.5, 0.05, 'Average Time (us)', ha='center', va='center')
    # 显示网格
    # for ax in axs:
    #     ax.grid(True)

    # 调整子图间距
    plt.tight_layout()

    # 展示图形
    plt.show()
    plt.savefig(f'./openfhe.pdf', format='pdf')

x1 = [1,2,4,8,16]
y1_1 = [24.5,24.4,24.475,24.5,24.5625]
y1_2 = [10.27,5.28,2.76,1.60375,1.03125]

y2_1 = [52.6,53,53,52.875,53.75]
y2_2 = [10.79,5.615,3.3125,2.1875,1.745625]

y3_1 = [113,113,112.75,114.75,116.75]
y3_2 = [11.94,6.865,4.7775,3.65625,3.06]

#y4_1 = [512,516,525.5,531.375,533.5]
y4_1 = y3_2
y4_2 = [21.7,16.785,13.68,11.74,10.61]

#draw_graph(x1,y1_1,x1,y1_2,x1,y2_1,x1,y2_2,x1,y3_1,x1,y3_2,x1,y4_1,x1,y4_2,"cpu 12","gpu 12","cpu 13","gpu 13","cpu 14","gpu 14","cpu 16","gpu 16","gpu vs cpu")
draw_23bar()