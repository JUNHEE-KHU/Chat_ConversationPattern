# Generated by Django 4.1.2 on 2023-06-08 13:10

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0007_userformalinformal_bebefore_informal_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='userformalinformal',
            name='bebefore_formal',
        ),
        migrations.RemoveField(
            model_name='userformalinformal',
            name='bebefore_informal',
        ),
        migrations.RemoveField(
            model_name='userformalinformal',
            name='before_formal',
        ),
        migrations.RemoveField(
            model_name='userformalinformal',
            name='before_informal',
        ),
    ]
