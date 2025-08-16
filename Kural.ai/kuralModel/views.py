from django.shortcuts import render
from django.http import HttpResponse


# Create your views here.
def Test(request):
    return render(request,'home.html')
def Input(request):
    if request.method == 'POST':
        name = request.POST.get('name')

        return render(request , 'result.html',{'name':name})
    return render(request , 'input.html')