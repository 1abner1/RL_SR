import tkinter as tk
from tkinter import messagebox

# 创建窗口
root = tk.Tk()
root.geometry("600x600")
root.title('GUI Example')

# 创建标签
label = tk.Label(root, text='Hello, World!', font=('Arial', 16))
label.pack(pady=20)

# 创建按钮
def click():
    messagebox.showinfo('Message', 'You clicked the button!')

button = tk.Button(root, text='Click me', command=click)
button.pack(pady=10)

# 运行窗口
root.mainloop()
