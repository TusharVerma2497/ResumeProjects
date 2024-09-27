import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
from PIL import ImageTk, ImageFilter
from tkinter import messagebox

Leftcanvas=None
Rightcanvas=None
Middlecanvas=None
mouseX=None
mouseY=None
turn=0
numberOfAnchorPoints=None
totalRequiredPoints=None
messageboxDisplayed=False

class CanvasWithContinuousUpdate(tk.Canvas):
    def __init__(self, parent, points, id, img, blurImg, **kwargs):
        tk.Canvas.__init__(self, parent, **kwargs)
        self.points=points
        self.img=img
        self.blurImg=blurImg
        self.id=id
        self.after(200, self.update_canvas)

    def update_canvas(self):
        global messageboxDisplayed
        # print('working')
        if(turn==self.id or totalRequiredPoints==0):
            self.create_image(0,0, image=self.img, anchor=tk.NW)
        else:
            self.create_image(0,0, image=self.blurImg, anchor=tk.NW)

        for i in range(len(self.points)):
            x1=self.points[i][1]+2
            y1=self.points[i][0]+2
            x2=self.points[i][1]-2
            y2=self.points[i][0]-2
            self.create_oval(x1, y1, x2, y2, outline="Red", width=2)
            smalX=x1 if x1<x2 else x2
            smalY =y1 if y1<y2 else y2
            self.create_text(smalX+12, smalY+12, text=str(i+1), font=('Helvetica','11','bold'), fill="Red")

        if(totalRequiredPoints==0 and messageboxDisplayed==False):
                messagebox.showinfo("Notice", "All the needed points have been selected\nKindly close the window!!")
                messageboxDisplayed=True
        self.after(200, self.update_canvas)




def get_swatches(source1, source2, target, n):
    global Leftcanvas
    global Rightcanvas
    global Middlecanvas
    global totalRequiredPoints
    global numberOfAnchorPoints

    numberOfAnchorPoints=n
    totalRequiredPoints=n*3
    root=tk.Tk()

    #creating title
    root.title('Select Anchor Points')

    # setting the geometry of window and fixing it
    originalSource1Shape=source1.size
    originalSource2Shape=source2.size
    originalTargetShape=target.size
    # print(originalSource1Shape)

    w=1000
    h=500
    root.geometry(f'{w}x{h}')
    root.resizable(width=False, height=False)

    
    # creating two canvas one left and one right for source and target image
    # blurSource1=source1.filter(ImageFilter.GaussianBlur(radius = 2))
    blurSource1=source1.convert('L')
    source1 = source1.resize((int(w/3),int((h))))
    blurSource1=blurSource1.resize((int(w/3),int((h))))
    source1 = ImageTk.PhotoImage(source1)
    blurSource1=ImageTk.PhotoImage(blurSource1)
    Leftcanvas=CanvasWithContinuousUpdate(root, points=[], id=0, img=source1, blurImg=blurSource1, width=(w)/3, height=(h))
    Leftcanvas.pack(side='left',fill='y')


    # blurSource2=source2.filter(ImageFilter.GaussianBlur(radius = 2))
    blurSource2=source2.convert('L')
    source2 = source2.resize((int(w/3),int((h))))
    blurSource2=blurSource2.resize((int(w/3),int((h))))
    source2 = ImageTk.PhotoImage(source2)
    blurSource2=ImageTk.PhotoImage(blurSource2)
    Middlecanvas=CanvasWithContinuousUpdate(root, points=[], id=1, img=source2, blurImg=blurSource2, width=(w)/3, height=(h))
    Middlecanvas.pack(side='left',fill='y')
    
    
    # blurTarget=target.filter(ImageFilter.GaussianBlur(radius = 2))
    blurTarget=target.convert('L')
    target= target.resize((int((w)/3),int((h))))
    blurTarget=blurTarget.resize((int((w)/3),int((h))))
    target = ImageTk.PhotoImage(target)
    blurTarget=ImageTk.PhotoImage(blurTarget)
    Rightcanvas=CanvasWithContinuousUpdate(root, points=[], id=2, img=target,  blurImg=blurTarget, width=(w)/3, height=(h))
    Rightcanvas.pack(side='left',fill='y')

    # # binding the mouse click event with the canvas
    Leftcanvas.bind("<Button-1>", on_leftCanvas_mouse_click)
    Middlecanvas.bind("<Button-1>", on_middleCanvas_mouse_click)
    Rightcanvas.bind("<Button-1>", on_rightCanvas_mouse_click)

    root.mainloop()
    
    #changing the range back to the original 
    LeftPoints=changeRange(Leftcanvas.points, w/3, h, originalSource1Shape[0], originalSource1Shape[1])
    MiddlePoints=changeRange(Middlecanvas.points, w/3, h, originalSource2Shape[0], originalSource2Shape[1])
    RightPoints=changeRange(Rightcanvas.points, w/3, h, originalTargetShape[0], originalTargetShape[1])

    return [LeftPoints, MiddlePoints, RightPoints]



def on_leftCanvas_mouse_click(event):
    global mouseX
    global mouseY
    global Leftcanvas
    global turn
    global totalRequiredPoints
    global numberOfAnchorPoints

    if(len(Leftcanvas.points) < numberOfAnchorPoints and turn==0):
        mouseX=event.x
        mouseY=event.y
        Leftcanvas.points.append([mouseY,mouseX])
        turn=(turn+1)%3
        totalRequiredPoints-=1


def on_middleCanvas_mouse_click(event):
    global mouseX
    global mouseY
    global Middlecanvas
    global turn
    global totalRequiredPoints
    global numberOfAnchorPoints

    if(len(Middlecanvas.points) < numberOfAnchorPoints and turn==1):
        mouseX=event.x
        mouseY=event.y
        Middlecanvas.points.append([mouseY,mouseX])
        turn=(turn+1)%3
        totalRequiredPoints-=1


def on_rightCanvas_mouse_click(event):
    global mouseX
    global mouseY
    global Rightcanvas
    global turn
    global totalRequiredPoints
    global numberOfAnchorPoints

    if(len(Rightcanvas.points) < numberOfAnchorPoints and turn==2):
        mouseX=event.x
        mouseY=event.y
        Rightcanvas.points.append([mouseY,mouseX])
        turn=(turn+1)%3
        totalRequiredPoints-=1

        

def changeRange(list, oldW, oldH, newW, newH):
    for i in range(len(list)):
        list[i]= (np.interp(list[i][0], (0, oldH), (0, newH)).clip(0,newH).astype(np.int32),
                  np.interp(list[i][1], (0, oldW), (0, newW)).clip(0,newW).astype(np.int32)
                  )
    return list