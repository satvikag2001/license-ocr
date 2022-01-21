from django.db import models
from django.contrib.auth.models import AbstractBaseUser
class Car(models.Model):
    #REQUIRED_FIELDS = ('cars',)
        
    #name = models.CharField(max_length=50)
    cars = models.ImageField(upload_to='images/')
    #cars = models.ImageField(upload_to='cars/')