# coding=utf-8

import os

import pymysql
import xlwt

cfg_username = "xxx"  # 用户名
cfg_password = "55556"  # 连接密码
cfg_url = "11.11.11.13"  # 连接地址
cfg_database = "exec"  # 数据库名

cfg_db_list = ["AAA", "VVV", "CCC"]
fname = 'database_doc.xls'


# 列出所有的表
def list_table(db_name):
    # 连接数据库
    db = pymysql.Connect(
        host=cfg_url,
        port=3307,
        user=cfg_username,
        passwd=cfg_password,
        db=db_name,
        charset='utf8'
    )
    cursor = db.cursor()
    cursor.execute("show tables")
    table_list = [tuple[0] for tuple in cursor.fetchall()]
    db.close()
    return table_list


# 查询所有字段
def list_col(db_name, tabls_name):
    # 连接数据库
    db = pymysql.Connect(
        host=cfg_url,
        port=3307,
        user=cfg_username,
        passwd=cfg_password,
        db=db_name,
        charset='utf8'
    )
    cursor = db.cursor()
    sql = r"SELECT COLUMN_KEY 主键,COLUMN_NAME 列名,DATA_TYPE 字段类型,CHARACTER_MAXIMUM_LENGTH 长度,IS_NULLABLE 是否为空,COLUMN_DEFAULT 默认值,COLUMN_COMMENT 备注 FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema = '{}' AND table_name ='{}'".format( db_name, tabls_name)
    cursor.execute(sql)
    rows = cursor.fetchall()
    db.close()
    return rows


# 设置表格样式
def set_style(name, height, bold=False):
    # 设置字体
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    # 设置边框
    borders = xlwt.Borders()
    # 细实线:1，小粗实线:2，细虚线:3，中细虚线:4，大粗实线:5，双线:6，细点虚线:7
    # 大粗虚线:8，细点划线:9，粗点划线:10，细双点划线:11，粗双点划线:12，斜点划线:13
    borders.left = 1
    borders.right = 1
    borders.top = 1
    borders.bottom = 1
    style.borders = borders

    return style


# 写Excel
def write_excel(wb, sheet_name):
    f_sheet = wb.add_sheet(sheet_name, cell_overwrite_ok=True)
    f_sheet.col(1).width = 256*20
    f_sheet.col(2).width = 256*10
    f_sheet.col(3).width = 256*10
    f_sheet.col(4).width = 256*10
    f_sheet.col(5).width = 256*25
    f_sheet.col(6).width = 256*40

    tables = list_table(sheet_name)
    dex = 0
    for t in tables:
        if dex != 0:
            dex = dex+3
            f_sheet.write(dex, 0, " ", set_style('Verdana', 200, True))

        # 写第一行
        f_sheet.write(0+dex, 0, "Table", set_style('Verdana', 200, True))
        f_sheet.write(0+dex, 1, t, set_style('Verdana', 200, True))
        f_sheet.write(0+dex, 2, " ", set_style('微软雅黑', 200))
        f_sheet.write(0+dex, 3, " ", set_style('微软雅黑', 200))
        f_sheet.write(0+dex, 4, " ", set_style('微软雅黑', 200))
        f_sheet.write(0+dex, 5, " ", set_style('微软雅黑', 200))
        f_sheet.write(0+dex, 6, " ", set_style('微软雅黑', 200))
        # 写第二行
        f_sheet.write(1+dex, 0, "键", set_style('微软雅黑', 200, True))
        f_sheet.write(1+dex, 1, "字段", set_style('微软雅黑', 200, True))
        f_sheet.write(1+dex, 2, "数据类型", set_style('微软雅黑', 200, True))
        f_sheet.write(1+dex, 3, "长度", set_style('微软雅黑', 200, True))
        f_sheet.write(1+dex, 4, "是否为空", set_style('微软雅黑', 220, True))
        f_sheet.write(1+dex, 5, "默认值", set_style('微软雅黑', 220, True))
        f_sheet.write(1+dex, 6, "注释", set_style('微软雅黑', 200, True))

        rows = list_col(sheet_name, t)
        row_list = [(tuple[0], tuple[1], tuple[2], tuple[3],
                     tuple[4], tuple[5], tuple[6]) for tuple in rows]
        row_num = len(row_list)
        # 写表的字段数据
        for i in range(row_num):
            for k in range(7):
                t_row = row_list[i]
                if k == 2:
                    f_sheet.write(
                        2+dex+i, k, (t_row[k]).upper(), set_style('微软雅黑', 200))
                else:
                    f_sheet.write(2+dex+i, k, t_row[k], set_style('微软雅黑', 200))

        dex = dex+row_num


# 生成数据库文档
def db_doc():
    wb = xlwt.Workbook(encoding='utf-8', style_compression=2)
    for db in cfg_db_list:
        print(db, "start generating......")
        write_excel(wb, db)
        print(db, "finished",
              "\n=========================================================")

    if os.path.exists(fname):
        os.remove(fname)
    wb.save(fname)

    print("Document has been generated")


if __name__ == "__main__":

    db_doc()