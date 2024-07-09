import os
from streamlit_option_menu import option_menu  #
import streamlit as st
from chat import chat_page, chat_page_v2
from rag import rag_page

# 主程序入口
if __name__ == '__main__':
    st.set_page_config(
        "ClinicaAIBrain",  # 设置标题
    )

    pages = {
        "对话": {
            "icon": "chat",  # 图标为 "chat"
            "func": chat_page_v2,  # 对应的函数为 chat_page
        },
        "知识库管理": {
            "icon": "hdd-stack",  # 图标为 "hdd-stack"
            "func": rag_page,  # 对应的函数为 rag_page
        },
    }

    # 在侧边栏中设置内容
    with st.sidebar:
        st.image(
            os.path.join(
                "img",
                "1.jpg"  # 显示 img 文件夹中的 1.jpg 图片
            ),
            use_column_width=True  # 使用列宽显示图片
        )

        options = list(pages)  # 获取页面的名称列表
        icons = [x["icon"] for x in pages.values()]  # 获取页面的图标列表

        default_index = 0  # 设置默认选项的索引为 0
        selected_page = option_menu(
            "",  # 设置菜单标题为空
            options=options,  # 设置菜单选项
            icons=icons,  # 设置菜单图标
            default_index=default_index,  # 设置默认选中的页面索引
        )

    # 如果选中的页面在 pages 字典中
    if selected_page in pages:
        pages[selected_page]["func"]()  # 执行选中页面对应的函数
