from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
#from django.shortcuts import render_to_response
#from django.template import RequestContext
#from django.http import HttpResponseRedirect
#from django.core.urlresolvers import reverse
from .forms import *
#global name_of_file
# our home page view
def a(request):
    return render(request,'a.html')
def home(request):
    #print("---------------------------------------first-------------------------------")
    #print(request.method) 
    if request.method == "POST":
        form = CarForm( request.POST, request.FILES)
        #name_of_file = request.FILES["cars"]
        #print("before save")
        if form.is_valid():
            form.save()     
            #print("------------------ahhhh",form)

            return redirect('result')
            #img_object = form.instance
            #return render(request, 'index.html', {'form' : form, 'img_obj' : img_object})

    else:
        #print("-------------------error-------------------------------------")
        form = CarForm()
    #return render(request, 'hotel_image_form.html', {'form' : form})   
    return render(request, 'index.html',{'form':form})

# custom method for generating predictions
from license_plate_web import getPredictions
        
import os
# our result page view
def result(request):
    path = "media/images/"
    img_list = os.listdir(path)  
    #print(img_list)
    img = path+img_list[0]
    #print("--------------------ohhhh--------------",img)
    result = getPredictions.getPredictions(img)
    os.remove(img)
    return render(request, 'result.html', {'result':result})
