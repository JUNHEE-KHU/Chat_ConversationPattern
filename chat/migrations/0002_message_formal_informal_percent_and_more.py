# Generated by Django 4.1.2 on 2023-06-05 06:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='message',
            name='formal_informal_percent',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='message',
            name='formal_informal_which',
            field=models.TextField(null=True),
        ),
    ]
