from django.db import models
from datetime import datetime

# Create your models here.
class Room(models.Model):
    name = models.CharField(max_length=1000)
class Message(models.Model):
    value = models.CharField(max_length=1000000)
    date = models.DateTimeField(default=datetime.now, blank=True)
    user = models.CharField(max_length=1000000)
    room = models.CharField(max_length=1000000)
    formal_informal_which = models.TextField(null=True)
    formal_informal_percent = models.IntegerField(null=True)
class User(models.Model):
    room = models.CharField(max_length=1000000)
    user = models.CharField(max_length=1000000)
    voca = models.CharField(max_length=1000000)
    count_var = models.IntegerField()
class UserFormalInformal(models.Model): 
    room = models.CharField(max_length=1000000)
    user = models.CharField(max_length=1000000)
    formal_count = models.IntegerField()
    informal_count = models.IntegerField()
    formal_percent_avg = models.IntegerField()
    # before_formal = models.BooleanField(default=0)
    # bebefore_formal = models.BooleanField(default=0)
    # before_informal = models.BooleanField(default=0)
    # bebefore_informal = models.BooleanField(default=0)
class Warning(models.Model):
    formal_warning_count = models.IntegerField(default=1, blank=True, null=True)
    voca_warning_count = models.IntegerField(default=1, blank=True, null=True)