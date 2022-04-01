import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import scrolledtext
import time
import random
import math

window = tk.Tk()
window.title("Multilayer Perceptron")

select_button = object
train_button = object
set_button = object
reset_button = object
file_label = object
learning_rate_label = object
epochs_label = object
learn_rate_entry = object
epochs_entry = object
canvas = object
output_text = object
result_line = object

filename = ""
dataset = list()
dataclass = list()
learning_rate = 0.1
epochs = 100
max_accuracy = 1
canvas_width = 500
canvas_height = 500
large_ratio = 1000
point_radius = 0.02
border = 50
center_dx, center_dy = 0, 0

class Perceptron():
    def __init__(self, dimension, learning_rate):
        self.learning_rate = learning_rate
        self.dataclass = dataclass
        self.w = [[random.uniform(-1, 1) for i in range(dimension + 1)]  for j in range(2)]
        self.w.append([random.uniform(-1, 1) for i in range(3)])

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def predict(self, data):
        v, y = list(), list()
        for j in range(len(self.w)):
            v.append(0)
            for i in range(len(self.w[j])):
                if i == 0:
                    v[j] += self.w[j][i] * -1
                elif j < 2:
                    v[j] += self.w[j][i] * data[i - 1]
                else:
                    v[j] += self.w[j][i] * y[i - 1]
            y.append(self.sigmoid(v[j]))
        return y

    def calculate_correct(self, correct, data, data_label, y):
        for i in range(len(data_label)):
            if i == 0 and y[-1] < (1 / len(data_label)) and data[-1] == data_label[0]:
                correct += 1
                break
            elif i == (len(data_label) - 1) and (
                    y[-1] >= ((len(data_label) - 1) / len(data_label)) and data[-1] == data_label[-1]):
                correct += 1
                break
            elif not i == 0 and not i == (len(data_label) - 1) and y[-1] >= (i / len(data_label)) and y[-1] < ((i + 1) / len(data_label)) and data[-1] == data_label[i]:
                correct += 1
                break
        return correct

    def calculate_accuracy(self, dataset, data_label):
        correct = 0
        rmse = 0
        for data in dataset:
            y = self.predict(data)
            rmse += (data[-1] - y[-1]) ** 2
            correct = self.calculate_correct(correct, data, data_label, y)
        rmse = (rmse / len(dataset)) ** (1 / 2)
        accuracy = correct / len(dataset)
        return rmse, accuracy

    def plot(self):
        min_x, max_x, min_y, max_y = 999, -999, 999, -999
        dataset_2 = list()
        for data in dataset:
            y = self.predict(data)
            y[-1] = data[-1]
            min_x, max_x, min_y, max_y = min(min_x, y[0]), max(max_x, y[0]), min(min_y, y[1]), max(max_y, y[1])
            dataset_2.append(y)
        adjustRatio(min_x, max_x, min_y, max_y)
        canvas.delete("all")
        for (x, y, types) in dataset_2:
            drawPoint(x, y, self.data_label.index(types))
        drawLine(self.w)

    def train(self, dataset, epochs):
        total_time = 0
        data_label = list()
        random.shuffle(dataset)
        train_dataset = dataset[:len(dataset) * 2 // 3].copy()
        test_dataset = dataset[len(dataset) * 2 // 3:].copy()

        # normalize the class to range of 0~1
        for i in range(len(self.dataclass)):
            data_label.append(i / (len(self.dataclass)) +
                              (1 / len(self.dataclass) / 2))
        for i in range(len(train_dataset)):
            for j in range(len(self.dataclass)):
                if train_dataset[i][-1] == self.dataclass[j]:
                    train_dataset[i][-1] = data_label[j]
        for i in range(len(test_dataset)):
            for j in range(len(self.dataclass)):
                if test_dataset[i][-1] == self.dataclass[j]:
                    test_dataset[i][-1] = data_label[j]

        for n in range(epochs):
            start_time = time.time()

            for train_data in train_dataset:
                y = self.predict(train_data)
                delta = [0 for i in range(len(self.w))]
                delta[-1] = (train_data[-1] - y[-1]) * y[-1] * (1 - y[-1])
                for j in range(len(self.w[-1]) - 1):
                    delta[j] = y[j] * (1 - y[j]) * delta[-1] * self.w[-1][j + 1]
                for j in range(len(self.w)):
                    for i in range(len(self.w[j])):
                        if i == 0:
                            self.w[j][i] += self.learning_rate * -1 * delta[j]
                        elif j < 2:
                            self.w[j][i] += self.learning_rate * train_data[i - 1] * delta[j]
                        else:
                            self.w[j][i] += self.learning_rate * y[i - 1] * delta[j]

            train_rmse, train_accuracy = self.calculate_accuracy(train_dataset, data_label)
            test_rmse, test_accuracy = self.calculate_accuracy(train_dataset, data_label)

            output_text.insert(tk.END, "Epoch {}:\ttrain acc  = {:.6f}: test acc  = {:.6f} ({:.6f} sec/epoch)\n".format(n + 1, train_accuracy, test_accuracy, time.time() - start_time))
            output_text.insert(tk.END, "\t" + " "*(len(str(n + 1))) + "train RMSE = {:.6f}: test RMSE = {:.6f}\n".format(train_rmse, test_rmse))
            total_time += time.time() - start_time
            if not max_accuracy == 0 and train_accuracy >= max_accuracy:
                break

        self.data_label = data_label
        if len(dataclass) == 2:
            self.plot()
        output_text.insert(tk.END, "\nw = {}\n\nTotal time: {:.6f} sec".format(self.w, total_time))
        output_text.see(tk.END)

def training():
    global dataset, learning_rate, epochs
    output_text.config(state="normal")
    output_text.delete(1.0, "end")
    perceptron = Perceptron(len(dataset[0][:-1]), learning_rate)
    perceptron.train(dataset, epochs)
    output_text.config(state="disabled")

def scaledX(x):
    return x * large_ratio + canvas_width // 2

def scaledY(y):
    return -y * large_ratio + canvas_height // 2

def adjustRatio(min_x, max_x, min_y, max_y):
    global large_ratio, point_radius, center_dx, center_dy
    large_ratio = 1000
    point_radius = 1 / large_ratio * 2
    center_dx, center_dy = (max_x + min_x) / 2, (max_y + min_y) / 2
    while large_ratio > 1 and (abs(scaledX(max_x - center_dx)) >= canvas_width - border or abs(scaledX(min_x - center_dx)) >= canvas_width - border or abs(scaledY(max_y - center_dy)) >= canvas_height - border or abs(scaledY(min_y - center_dy)) >= canvas_height - border):
        large_ratio = large_ratio - 1
        point_radius = 1 / large_ratio * 2

def drawLine(w):
    global result_line
    if w[-1][1] != 0:
        x1 = -100
        y1 = (w[-1][0] - (w[-1][1] * x1)) / w[-1][2]
        x2 = 100
        y2 = (w[-1][0] - (w[-1][1] * x2)) / w[-1][2]

    canvas.delete(result_line)
    result_line = canvas.create_line(scaledX(x1 - center_dx), scaledY(y1 - center_dy),
                        scaledX(x2 - center_dx),scaledY(y2 - center_dy), fill='red')
    canvas.update_idletasks()

def drawPoint(x, y, types):
    colors = ['blue', 'orange', 'green', 'red', 'pink', 'brown', 'purple']
    canvas.create_oval(scaledX(x - center_dx - point_radius), scaledY(y - center_dy + point_radius),
                       scaledX(x - center_dx + point_radius), scaledY(y - center_dy - point_radius), fill=colors[types])
    canvas.update_idletasks()

def readFile(file):
    global dataset, dataclass
    min_x, max_x, min_y, max_y = 999, -999, 999, -999

    f = open(file, "r")
    for i in f.readlines():
        line = list(map(float, i.split()))
        dataset.append(line)

    if len(dataset[0][:-1]) == 2:
        for (x, y, types) in dataset:
            min_x, max_x, min_y, max_y = min(min_x, x), max(max_x, x), min(min_y, y), max(max_y, y)
            if types not in dataclass:
                dataclass.append(types)
        dataset.sort()
        adjustRatio(min_x, max_x, min_y, max_y)
        for (x, y, types) in dataset:
            drawPoint(x, y, dataclass.index(types))
    f.close()

def selectDataset():
    global filename
    resetState()
    file = filedialog.askopenfilename(title="Select file", filetypes=(("text files (*.txt)", "*.txt"), ("all files", "*.*")))
    if file:
        readFile(file)
        filename = file.split("/")[-1]
        file_label.config(text="{}".format(filename))
        train_button.config(state="normal")
        set_button.config(state="normal")
        reset_button.config(state="normal")
        learn_rate_entry.config(state="normal")
        epochs_entry.config(state="normal")
    else:
        tk.messagebox.showinfo(title="Notice", message="No file selected")

def setParameter():
    global learning_rate, epochs

    if learn_rate_entry.get() or epochs_entry.get():
        pass
    else:
        tk.messagebox.showinfo(title="Notice", message="Please input parameters")
        return

    if learn_rate_entry.get():
        learning_rate = float(learn_rate_entry.get())
        if learning_rate > 0:
            learning_rate_label.config(text="learning rate = {}".format(learning_rate))
            learn_rate_entry.delete(0, "end")
        else:
            tk.messagebox.showinfo(title="Notice", message="Please input value > 0")

    if epochs_entry.get():
        epochs = int(epochs_entry.get())
        if epochs > 0:
            epochs_label.config(text="epochs = {}".format(epochs))
            epochs_entry.delete(0, "end")
        else:
            tk.messagebox.showinfo(title="Notice", message="Please input value > 0")

def resetState():
    global dataset, learning_rate, epochs, large_ratio, point_radius, dataclass
    dataset = list()
    dataclass = list()
    learning_rate = 0.1
    epochs = 100
    large_ratio = 1000
    point_radius = 0.02
    learning_rate_label.config(text="learning rate = {}".format(learning_rate))
    epochs_label.config(text="epochs = {}".format(epochs))
    train_button.config(state="disabled")
    set_button.config(state="disabled")
    reset_button.config(state="disabled")
    file_label.config(text="{}".format(""))
    learn_rate_entry.delete(0, "end")
    learn_rate_entry.config(state="disabled")
    epochs_entry.delete(0, "end")
    epochs_entry.config(state="disabled")
    canvas.delete("all")
    output_text.config(state="normal")
    if output_text.get(1.0, tk.END):
        output_text.delete(1.0, "end")
    output_text.config(state="disabled")

def GUI():
    global select_button, train_button, set_button, reset_button, canvas, output_text
    global file_label, learning_rate_label, epochs_label, learn_rate_entry, epochs_entry

    select_button = tk.Button(window, text="select dataset", command=selectDataset, width=15)
    select_button.grid(row=0, column=0)
    train_button = tk.Button(window, text="start training", command=training, state="disabled", width=15)
    train_button.grid(row=1, column=0)
    set_button = tk.Button(window, text="set parameter", command=setParameter, state="disabled", width=15)
    set_button.grid(row=2, column=0)
    reset_button = tk.Button(window, text="clear all", command=resetState, state="disabled", width=15)
    reset_button.grid(row=5, column=0)

    file_label = tk.Label(window, text="{}".format(filename), width=20)
    file_label.grid(row=0, column=1)
    learning_rate_label = tk.Label(window, text="learning rate = {}".format(learning_rate), width=15)
    learning_rate_label.grid(row=3, column=0)
    epochs_label = tk.Label(window, text="epochs = {}".format(epochs), width=15)
    epochs_label.grid(row=4, column=0)

    learn_rate_entry = tk.Entry(window, state="disabled", width=15)
    learn_rate_entry.grid(row=3, column=1)
    epochs_entry = tk.Entry(window, state="disabled", width=15)
    epochs_entry.grid(row=4, column=1)

    canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='white')
    canvas.grid(row=0, column=3, rowspan=6)

    output_text = scrolledtext.ScrolledText(window, state="disabled", height=38)
    output_text.grid(row=0, column=4, rowspan=6)

    tk.mainloop()

if __name__ == "__main__":
    GUI()