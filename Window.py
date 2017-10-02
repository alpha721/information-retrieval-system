from Tkinter import *
#import search

window = Tk()
window.title ("IR")

l=Label (window, text="GIVE A QUERY AND CLICK ON THE BUTTON:")

entry = Entry(window, width = 75, bg = "light blue")
entry.grid(row=1,column=0,sticky=W)

b=Button (window, text="Submit",command="search.main()",bg="blue")
b.grid(row=2,column=0,sticky=W)

output = Text(window,width=75,height=6,background="white",foreground="black")
output.grid(row=3,column=0,sticky=W)
l.pack()
entry.pack()
b.pack()
output.pack()
window.mainloop()
