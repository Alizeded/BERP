import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_correlation_matrix(
    correlation_values,
    labels=None,
    color_theme="ocean",
    output_path="correlation_matrix.pdf",
):
    plt.rcParams["text.usetex"] = False

    # 設置字體大小
    plt.rcParams["xtick.labelsize"] = 26
    plt.rcParams["ytick.labelsize"] = 26

    # 設置對角線為None
    # mask = np.eye(len(correlation_values), dtype=bool)
    # masked_corr = np.ma.array(correlation_values, mask=mask)

    # 顏色主題
    color_themes = {
        "ocean": [
            "#ffffff",
            "#ecfeff",
            "#cffafe",
            "#a5f3fc",
            "#67e8f9",
            "#22d3ee",
            "#06b6d4",
            "#0891b2",
            "#0e7490",
            "#155e75",
        ],
        "purple": [
            "#ffffff",
            "#f3e8ff",
            "#e9d5ff",
            "#d8b4fe",
            "#c084fc",
            "#a855f7",
            "#9333ea",
            "#7e22ce",
            "#6b21a8",
            "#581c87",
        ],  # 優雅紫色
    }

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

    # 創建自定義顏色映射
    custom_cmap = LinearSegmentedColormap.from_list("custom", color_themes[color_theme])

    # 創建熱圖
    ax = sns.heatmap(
        correlation_values,
        cmap=custom_cmap,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Weak corr.               Strong corr.", "ticks": []},
        # mask=mask,
        ax=ax,
    )

    cbar = ax.collections[0].colorbar
    cbar.set_label(
        "Weak corr.               Strong corr.", size=26
    )  # 在這裡設置字體大小

    # 在創建熱圖後添加以下代碼
    # 添加外邊框
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    # 設置邊框顏色和寬度
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(2)

    # 如果想要更精細的控制，可以分別設置
    ax.spines["top"].set_linewidth(1.5)  # 上邊框
    ax.spines["right"].set_linewidth(1.5)  # 右邊框
    ax.spines["bottom"].set_linewidth(1.5)  # 下邊框
    ax.spines["left"].set_linewidth(1.5)  # 左邊框

    # 設置對角線單元格的顏色為白色
    # for i in range(len(correlation_values)):
    #     ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="white"))

    # 美化邊框和網格
    ax.collections[0].set_linewidth(0.5)
    ax.collections[0].set_edgecolor("white")

    # # 設置標題和標籤
    # ax.set_title("Mutual Relatedness", pad=20, fontsize=18, fontweight="bold")
    # ax.set_xlabel("Room parameters", labelpad=10, fontsize=18)
    # ax.set_ylabel("Room parameters", labelpad=10, fontsize=18)

    # 設置刻度標籤的旋轉
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # 調整布局
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
        format="pdf",
        facecolor="white",
        edgecolor="none",
    )

    plt.show()


# 直接從列表創建numpy數組
correlation_values = np.array(
    [
        [1, 0.97, 0.71, 0.64, 0.91, 0.84, 0.0, 0.84, 0.0, 0.0],
        [0.97, 1, 0.69, 0.66, 0.86, 0.80, 0.0, 0.81, 0.95, 0.95],
        [0.71, 0.69, 1, 0.82, 0.75, 0.53, 0.45, 0.82, 0.95, 0.0],
        [0.64, 0.66, 0.82, 1, 0.69, 0.33, 0.45, 0.67, 0.0, 0.0],
        [0.91, 0.86, 0.75, 0.69, 1, 0.86, 0.0, 0.89, 0.0, 0.0],
        [0.84, 0.80, 0.53, 0.33, 0.86, 1, 0.0, 0.81, 0.0, 0.0],
        [0.0, 0.0, 0.45, 0.45, 0.0, 0.95, 1, 0.0, 0.0, 0.0],
        [0.84, 0.81, 0.82, 0.67, 0.89, 0.81, 0.0, 1, 0.0, 0.0],
        [0.0, 0.95, 0.95, 0.0, 0.0, 0.0, 0.45, 0.0, 1, 0.95],
        [0.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 1],
    ]
)

# 設置對角線為1
np.fill_diagonal(correlation_values, 1.0)

# 設置標籤
labels = [
    r"$STI$",
    r"$\%AL_{cons}$",
    r"$T_{60}$",
    r"$EDT$",
    r"$C_{80}$",
    r"$C_{50}$",
    r"$D_{50}$",
    r"$T_s$",
    r"$V$",
    r"$D$",
]

# 繪製矩陣
plot_correlation_matrix(
    correlation_values=correlation_values,
    labels=labels,
    color_theme="ocean",
    output_path="./data/Figure/mutual_relatedness.pdf",
)
