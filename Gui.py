from tkinter import Tk, Label, PhotoImage

root = Tk()
root.title("My GUI")

# Load and display an image
image = PhotoImage(file="path/to/image.png")
label = Label(root, image=image)
label.pack()

root.mainloop()
