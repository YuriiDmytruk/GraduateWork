from tkinter import *
from tkinter import ttk


def submit():
    cryptocurrency = cryptocurrency_combobox.get()
    start_time = start_time_entry.get()
    end_time = end_time_entry.get()
    selected_indicators = [indicator.get() for indicator,
                           var in indicator_checkboxes.items() if var.get() == 1]
    validation_data = int(validation_data_entry.get())
    historical_element = historical_element_entry.get()
    epochs = int(epochs_entry.get())

    # Вивід отриманих значень на консоль
    print("Криптовалюта:", cryptocurrency)
    print("Початковий час:", start_time)
    print("Кінцевий час:", end_time)
    print("Обрані показники:", selected_indicators)
    print("Кількість тренувальних даних:", validation_data)
    print("Історичний елемент:", historical_element)
    print("Кількість епох:", epochs)


root = Tk()
root.title("Вибір криптовалюти, показників та інших параметрів")

# Створюємо випадаючий список для вибору криптовалюти
cryptocurrency_combobox = ttk.Combobox(
    root, values=["Bitcoin", "Ethereum", "Litecoin"])
cryptocurrency_combobox.grid(row=0, column=0, padx=10, pady=10)

# Створюємо текстові поля для введення початкового та кінцевого часу
start_time_label = Label(root, text="Початковий час:")
start_time_label.grid(row=1, column=0, padx=10, pady=5)
start_time_entry = Entry(root)
start_time_entry.grid(row=1, column=1, padx=10, pady=5)

end_time_label = Label(root, text="Кінцевий час:")
end_time_label.grid(row=2, column=0, padx=10, pady=5)
end_time_entry = Entry(root)
end_time_entry.grid(row=2, column=1, padx=10, pady=5)

# Створюємо прапорці для вибору показників
indicators = ["Open", "High", "Low", "Close", "Volume", "Price"]
indicator_checkboxes = {}
for i, indicator in enumerate(indicators):
    var = IntVar()
    checkbox = Checkbutton(root, text=indicator, variable=var)
    checkbox.grid(row=i+3, column=0, padx=10, pady=2)
    indicator_checkboxes[indicator] = var

# Створюємо прапорці для вибору індикаторів
selected_indicators_label = Label(root, text="Оберіть індикатори:")
selected_indicators_label.grid(row=len(indicators)+3, column=0, padx=10, pady=5)

indicators = ["RSI", "EMA", "MACD", "ATR", "BollingerBands", "MFI", "OBV", "ADX"]
indicator_checkboxes = {}
for i, indicator in enumerate(indicators):
    var = IntVar()
    checkbox = Checkbutton(root, text=indicator, variable=var)
    checkbox.grid(row=i+len(indicators)+4, column=0, padx=10, pady=2)
    indicator_checkboxes[indicator] = var

# Створюємо поля для вводу кількості даних, історичного елемента та кількості епох
validation_data_label = Label(root, text="Кількість тренувальних даних:")
validation_data_label.grid(row=len(indicators)*2+6, column=0, padx=10, pady=5)
validation_data_entry = Entry(root)
validation_data_entry.grid(row=len(indicators)*2+6, column=1, padx=10, pady=5)

historical_element_label = Label(root, text="Історичний елемент:")
historical_element_label.grid(row=len(indicators)*2+7, column=0, padx=10, pady=5)
historical_element_entry = Entry(root)
historical_element_entry.grid(row=len(indicators)*2+7, column=1, padx=10, pady=5)

epochs_label = Label(root, text="Кількість епох:")
epochs_label.grid(row=len(indicators)*2+8, column=0, padx=10, pady=5)
epochs_entry = Entry(root)
epochs_entry.grid(row=len(indicators)*2+8, column=1, padx=10, pady=5)

# Кнопка для підтвердження вибору
submit_button = Button(root, text="Підтвердити", command=submit)
submit_button.grid(row=len(indicators)*2+9, column=0, padx=10, pady=10)

root.mainloop()
