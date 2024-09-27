import tkinter as tk
from PIL import ImageTk, Image
import numpy as np

Leftcanvas=None
Rightcanvas=None
dragging=False
mouseX=None
mouseY=None

class CanvasWithContinuousUpdate(tk.Canvas):
    def __init__(self, parent, MouseDragAllowed, swatch, img, **kwargs):
        tk.Canvas.__init__(self, parent, **kwargs)
        self.swatch=swatch
        self.img=img
        self.MouseDragAllowed=MouseDragAllowed
        self.after(500, self.update_canvas)

    def update_canvas(self):
        # print('working')
        self.create_image(0,0, image=self.img,anchor=tk.NW)
        if(dragging==False):
            for i in range(1,len(self.swatch),2):
                x1=self.swatch[i-1][0]
                y1=self.swatch[i-1][1]
                x2=self.swatch[i][0]
                y2=self.swatch[i][1]
                self.create_rectangle(x1, y1, x2, y2, outline="Red", width=3)
        else:
            if(self.MouseDragAllowed):
                for i in range(0,len(self.swatch)-1,2):
                    x1=self.swatch[i][0]
                    y1=self.swatch[i][1]
                    x2=self.swatch[i+1][0]
                    y2=self.swatch[i+1][1]
                    self.create_rectangle(x1, y1, x2, y2, outline="Red", width=3)
                self.create_rectangle(self.swatch[-1][0], self.swatch[-1][1], mouseX, mouseY, outline="Red", width=3)
            else:
                for i in range(1,len(self.swatch),2):
                    x1=self.swatch[i-1][0]
                    y1=self.swatch[i-1][1]
                    x2=self.swatch[i][0]
                    y2=self.swatch[i][1]
                    self.create_rectangle(x1, y1, x2, y2, outline="Red", width=3)
        self.after(500, self.update_canvas)





def get_swatches(source, target):
    global Leftcanvas
    global Rightcanvas
    root=tk.Tk()

    #creating title
    root.title('Select Swatches')

    #setting the geometry of window and fixing it
    originalSourceShape=source.size
    originalTargetShape=target.size
    w=1450
    h=750
    root.geometry(f'{w}x{h}')
    root.resizable(width=False, height=False)


    # creating two canvas one left and one right for source and target image
    source= source.resize((int((w-25)/2),int((h-50))))
    source = ImageTk.PhotoImage(source)
    Leftcanvas=CanvasWithContinuousUpdate(root, swatch=[], MouseDragAllowed=True, img=source, width=(w-25)/2, height=(h-50))
    Leftcanvas.pack(side='left',fill='y')
    
    
    target= target.resize((int((w-25)/2),int((h-50))))
    target = ImageTk.PhotoImage(target)
    Rightcanvas=CanvasWithContinuousUpdate(root, swatch=[], MouseDragAllowed=False, img=target, width=(w-25)/2, height=(h-50))
    Rightcanvas.pack(side='right',fill='y')

    # binding the mouse drag event with the canvas
    Leftcanvas.bind("<B1-Motion>", on_left_mouse_drag)
    Leftcanvas.bind("<ButtonRelease-1>", on_left_mouse_release)


    Rightcanvas.bind("<B1-Motion>", on_right_mouse_drag)
    Rightcanvas.bind("<ButtonRelease-1>", on_right_mouse_release)

    root.mainloop()

    #changing the range back to the original 
    leftSwatches=changeRange(Leftcanvas.swatch, (w-25)/2, h-50, originalSourceShape[0], originalSourceShape[1])
    RightSwatches=changeRange(Rightcanvas.swatch, (w-25)/2, h-50, originalTargetShape[0], originalTargetShape[1])
    
    return leftSwatches, RightSwatches




def on_left_mouse_drag(event):
    global dragging
    global mouseX
    global mouseY
    global Leftcanvas

    dragging=True
    mouseX=event.x
    mouseY=event.y
    # Print the current mouse position while dragging
    if(Leftcanvas.MouseDragAllowed):
        if(len(Leftcanvas.swatch)%2==0):
            Leftcanvas.swatch.append((event.x, event.y))

def on_left_mouse_release(event):
    global dragging
    global mouseX
    global mouseY
    global Leftcanvas
    global Rightcanvas

    dragging=False
    mouseX=None
    mouseY=None
    if(Leftcanvas.MouseDragAllowed):
        Leftcanvas.swatch.append((event.x, event.y))
        Leftcanvas.MouseDragAllowed=False
        Rightcanvas.MouseDragAllowed=True




def on_right_mouse_drag(event):
    global dragging
    global mouseX
    global mouseY
    global Rightcanvas

    dragging=True
    mouseX=event.x
    mouseY=event.y
    # Print the current mouse position while dragging
    if(Rightcanvas.MouseDragAllowed):
        if(len(Rightcanvas.swatch)%2==0):
            Rightcanvas.swatch.append((event.x, event.y))



def on_right_mouse_release(event):
    global dragging
    global mouseX
    global mouseY
    global Leftcanvas
    global Rightcanvas

    dragging=False
    mouseX=None
    mouseY=None
    if(Rightcanvas.MouseDragAllowed):
        Rightcanvas.swatch.append((event.x, event.y))
        Leftcanvas.MouseDragAllowed=True
        Rightcanvas.MouseDragAllowed=False



def changeRange(list, oldW, oldH, newW, newH):
    for i in range(len(list)):
        list[i]= (np.interp(list[i][1], (0, oldH), (0, newH)).clip(0,newH).astype(np.int32),
                  np.interp(list[i][0], (0, oldW), (0, newW)).clip(0,newW).astype(np.int32))
    return list
