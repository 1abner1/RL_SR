from tkinter import messagebox
import tkinter as tk

info_list = []
window = tk.Tk()
window.title('员工管理系统')
window.geometry('600x600')

staff_title = tk.Label(window, text='欢迎来到员工管理系统', bg='yellow', font=("黑体",15,"bold"))
staff_title.pack()


def add_staff_info():  # 添加

    def add_ok_info():  # 添加完成按钮

        new_info_dict = dict()

        nn = new_name.get()
        ns = new_sex.get()
        na = new_age.get()
        nb = new_branch.get()
        np = new_phone.get()

        new_info_dict['姓名'] = nn
        new_info_dict['性别'] = ns
        new_info_dict['年龄'] = na
        new_info_dict['部门'] = nb
        new_info_dict['手机号码'] = np

        info_list.append(new_info_dict)
        add_window.destroy()
        tk.messagebox.showinfo(title='提示', message='添加成功！')

    new_name = tk.StringVar()
    new_sex = tk.StringVar()
    new_age = tk.StringVar()
    new_branch = tk.StringVar()
    new_phone = tk.StringVar()

    add_window = tk.Toplevel(window)
    add_window.title('添加信息')
    add_window.geometry('600x600')

    tk.Label(add_window, text='姓      名：').place(x=20, y=10)
    entry_new_name = tk.Entry(add_window, textvariable=new_name)
    entry_new_name.place(x=80, y=10)

    tk.Label(add_window, text='性      别：').place(x=10, y=40)
    entry_new_name = tk.Entry(add_window, textvariable=new_sex)
    entry_new_name.place(x=80, y=40)

    tk.Label(add_window, text='年      龄：').place(x=10, y=70)
    entry_new_name = tk.Entry(add_window, textvariable=new_age)
    entry_new_name.place(x=80, y=70)

    tk.Label(add_window, text='部      门：').place(x=10, y=100)
    entry_new_name = tk.Entry(add_window, textvariable=new_branch)
    entry_new_name.place(x=80, y=100)

    tk.Label(add_window, text='手机号码：').place(x=10, y=130)
    entry_new_name = tk.Entry(add_window, textvariable=new_phone)
    entry_new_name.place(x=80, y=130)

    add_ok_button = tk.Button(add_window, text='完成', command=add_ok_info)
    add_ok_button.place(x=100, y=200)


def show_staff_info():  # 显示
    i = 20
    show_window = tk.Toplevel(window)
    show_window.title('显示员工列表')
    show_window.geometry('320x300')

    tk.Label(show_window, text='姓  名        性  别        年  龄        部  门        手 机 号 码',
             bg='yellow').pack()
    for temp_info in info_list:
        tk.Label(show_window, text=temp_info['姓名']).place(x=0, y=i)
        tk.Label(show_window, text=temp_info['性别']).place(x=70, y=i)
        tk.Label(show_window, text=temp_info['年龄']).place(x=130, y=i)
        tk.Label(show_window, text=temp_info['部门']).place(x=190, y=i)
        tk.Label(show_window, text=temp_info['手机号码']).place(x=260, y=i)
        i += 30


def del_staff_info():  # 删除

    def del_ok_info():  # 确认删除按钮

        dn = del_name.get()
        for i in info_list:
            if dn in i['姓名']:
                del info_list[info_list.index(i)]
                del_window.destroy()
                tk.messagebox.showinfo(title='提示', message='删除成功！')

    del_name = tk.StringVar()

    del_window = tk.Toplevel(window)
    del_window.title('删除信息')
    del_window.geometry('350x200')

    tk.Label(del_window, text='请输入要删除的员工姓名：').place(x=10, y=10)
    entry_del_name = tk.Entry(del_window, textvariable=del_name)
    entry_del_name.place(x=160, y=10)

    del_ok_button = tk.Button(del_window, text='确认删除', command=del_ok_info)
    del_ok_button.place(x=150, y=150)


def modify_staff_info():  # 修改

    def modify_ok_info():  # 确认修改按钮
        flag = 0
        on = old_name.get()
        for i in info_list:
            if on in i['姓名']:
                flag = 1
                info_list[info_list.index(i)]['姓名'] = new_name.get()
                info_list[info_list.index(i)]['性别'] = new_sex.get()
                info_list[info_list.index(i)]['年龄'] = new_age.get()
                info_list[info_list.index(i)]['部门'] = new_branch.get()
                info_list[info_list.index(i)]['手机号码'] = new_phone.get()
                modify_window.destroy()
                tk.messagebox.showinfo(title='提示', message='修改成功！')

        if flag == 0:
            tk.messagebox.showerror(title='警告', message='查无此人！')

    if len(info_list) == 0:
        tk.messagebox.showwarning(title='警告', message='员工列表为空！')
    else:
        old_name = tk.StringVar()

        new_name = tk.StringVar()
        new_sex = tk.StringVar()
        new_age = tk.StringVar()
        new_branch = tk.StringVar()
        new_phone = tk.StringVar()

        modify_window = tk.Toplevel(window)
        modify_window.title('修改信息')
        modify_window.geometry('350x300')

        tk.Label(modify_window, text='请输入要修改的员工姓名：').place(x=10, y=10)
        entry_old_name = tk.Entry(modify_window, textvariable=old_name)
        entry_old_name.place(x=160, y=10)

        tk.Label(modify_window, text='--------------------请输入要修改后员工信息--------------------').place(x=0, y=50)

        tk.Label(modify_window, text='姓  名：').place(x=10, y=80)
        entry_new_name = tk.Entry(modify_window, textvariable=new_name)
        entry_new_name.place(x=60, y=80)

        tk.Label(modify_window, text='性  别：').place(x=10, y=110)
        entry_new_name = tk.Entry(modify_window, textvariable=new_sex)
        entry_new_name.place(x=60, y=110)

        tk.Label(modify_window, text='年  龄：').place(x=10, y=140)
        entry_new_name = tk.Entry(modify_window, textvariable=new_age)
        entry_new_name.place(x=60, y=140)

        tk.Label(modify_window, text='部  门：').place(x=10, y=170)
        entry_new_name = tk.Entry(modify_window, textvariable=new_branch)
        entry_new_name.place(x=60, y=170)

        tk.Label(modify_window, text='部  门：').place(x=10, y=200)
        entry_new_name = tk.Entry(modify_window, textvariable=new_phone)
        entry_new_name.place(x=60, y=200)

        modify_ok_button = tk.Button(modify_window, text='确认修改', command=modify_ok_info)
        modify_ok_button.place(x=140, y=250)


def lookup_staff_info():  # 查看

    def lookup_ok_info():  # 查找按钮
        flag = 0
        ln = lookup_name.get()
        for i in info_list:
            if ln in i['姓名']:
                flag = 1
                staff_info = tk.Toplevel(lookup_window)
                staff_info.title('员工信息')
                staff_info.geometry('300x300')
                tk.Label(staff_info, text='姓      名：%s' % i['姓名']).place(x=10, y=10)
                tk.Label(staff_info, text='性      别：%s' % i['性别']).place(x=10, y=40)
                tk.Label(staff_info, text='年      龄：%s' % i['年龄']).place(x=10, y=70)
                tk.Label(staff_info, text='部      门：%s' % i['部门']).place(x=10, y=100)
                tk.Label(staff_info, text='手机号码：%s' % i['手机号码']).place(x=10, y=130)
                tk.Button(staff_info, text='关 闭', command=lookup_window.destroy).place(x=200, y=200)
                tk.Button(staff_info, text='继续查找', command=staff_info.destroy).place(x=50, y=200)

        if flag == 0:
            tk.messagebox.showerror(title='提示', message='查无此人！')

    if len(info_list) == 0:
        tk.messagebox.showwarning(title='警告', message='员工列表为空！')
    else:
        lookup_name = tk.StringVar()
        lookup_name.set('NONE')

        lookup_window = tk.Toplevel(window)
        lookup_window.title('查看信息')
        lookup_window.geometry('600x500')

        tk.Label(lookup_window, text='员工姓名：').place(x=10, y=10)
        entry_old_name = tk.Entry(lookup_window, textvariable=lookup_name)
        entry_old_name.place(x=80, y=10)

        modify_ok_button = tk.Button(lookup_window, text='查  找', command=lookup_ok_info)
        modify_ok_button.place(x=130, y=150)


add_button = tk.Button(window, text='1.添加员工信息', command=add_staff_info,height=2,width=30,font=("黑体",15,"bold"),bg="blue",fg="red")
# add_button.place(x=150,y=900)
add_button.pack(side = "left" )
del_button = tk.Button(window, text='2.删除员工信息', command=del_staff_info,height=2,width=20,font=("黑体",15,"bold"))
del_button.pack()
modify_button = tk.Button(window, text='3.修改员工信息', command=modify_staff_info,height=2,width=20,font=("黑体",15,"bold"))
modify_button.pack()
lookup_button = tk.Button(window, text='4.查找员工信息', command=lookup_staff_info,height=2,width=20,font=("黑体",15,"bold"))
lookup_button.pack()
show_button = tk.Button(window, text='5.显示员工信息', command=show_staff_info,height=2,width=20,font=("黑体",15,"bold"))
show_button.pack()
show_button = tk.Button(window, text='0.退  出  系  统', command=window.destroy,height=2,width=20,font=("黑体",15,"bold"))
show_button.pack()

window.mainloop()