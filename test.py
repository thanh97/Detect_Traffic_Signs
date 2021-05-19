from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

import numpy
#load the trained model to classify sign
from keras.models import load_model
model = load_model('./model/my_model.h5')

#dictionary to label all traffic signs class.
classes = { 0:'Bien bao cam o to',
            1:'Bien bao cam re trai',
            2:'Bien bao cam re phai',
            3:'Bien bao cam do xe',
            4:'Bien bao duong khong bang phang',
            5:'Bien bao huong phai di vong sang trai',
            6:'Bien bao cam di nguoc chieu',
            7:'Bien bao cam xe khach va xe tai',
            8:'Bien bao toc do toi da cho phep',
            9:'Bien bao cam dung xe va do xe',
            10:'Bien bao het tat ca cac lenh cam',
            11:'Bien bao cho ngoat nguy hiem',
            12:'Bien bao duong giao nhau',
            13:'Bien bao giao nhau duong khong uu tien',
            14:'Bien bao giao nhau duong uu tien',
            15:'Bien bao noi giao nhau co tin hieu den',
            16:'Bien bao nguoi di bo cat ngang',
            17:'Bien bao tre em',
            18:'Bien nguoi di xe dap cat ngang',
            19:'Bien bao nguy hiem khac',
            20:'Bien bao noi giao nhau chay theo vong xuyen',
            21:'Bien bao cho quay xe',
            22:'Bien bao duong di bo',
            23:'Bien bao benh vien',
            24:'Bien bao tram xang',
            25:'Bien bao phan lan'
             }
#
# #initialise GUI
top = Tk()
top.geometry('800x600')
top.title('He thong nhan dien bien bao giao thong')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30,30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    pred = model.predict_classes(image)[0]
    sign = classes[pred]
    print(sign)
    label.configure(foreground='#011638', text=sign)

def show_classify_button(file_path):
    classify_b=Button(top,text="Nhan dang",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Tai anh len",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="NHAN DIEN BIEN BAO GIAO THONG",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()